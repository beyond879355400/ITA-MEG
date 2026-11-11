#!/usr/bin/env python3
"""Bounded public Drive metadata/header probe; never saves model weights."""
import argparse
import concurrent.futures
import html
import http.cookiejar
import json
import re
import urllib.parse
import urllib.request

FILES = [
    ('F', 'appearance_feature_extractor.pth', '1YnmRePhPjtKY3Us8C0Mg-VInGVSVepB5'),
    ('M', 'motion_extractor.pth', '1VhcFZrTCpJvSpSuGVSPo3bH4XsGVw5oj'),
    ('G', 'spade_generator.pth', '1G6NKloq29ULCQeJAw-4qvH0nA-5oj6-T'),
    ('W', 'warping_module.pth', '1Pvh8foijP1jHwQxnP-676UZH86krdyOK'),
    ('DINO', 'pytorch_model.bin', '1L3HuaJNhd4fZ2E5SO2ZNxmwXpYmtXav0'),
    ('landmark', 'landmark.onnx', '1CjnT5pT1dKIE2SYuizBlhRge2fTohA4w'),
    ('buffalo', 'buffalo_l.zip', '1U6Tgh4Rshhr-vbedvkEr2i4J_yN-C7u9'),
]


def probe(row, header):
    name, filename, file_id = row
    result = {'component': name, 'expected_filename': filename, 'drive_id': file_id,
              'weight_file_saved': False, 'binary_bytes_read': 0}
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    try:
        url = 'https://drive.google.com/file/d/' + file_id + '/view'
        with opener.open(url, timeout=30) as response:
            typ = response.headers.get('Content-Type', '')
            if 'text/html' not in typ:
                raise RuntimeError('Viewer not HTML; no body read')
            body = response.read(524288).decode('utf8', 'replace')
        match = re.search(r'<title>(.*?)</title>', body, re.S)
        result['viewer_title'] = html.unescape(match.group(1)) if match else None
        if filename not in (result['viewer_title'] or ''):
            raise RuntimeError('Viewer filename mismatch or access gate')
        start = body.find('itemJson: ')
        if start >= 0:
            items, _ = json.JSONDecoder().raw_decode(body[start + len('itemJson: '):])
            result['item_numeric_fields'] = {str(i): v for i, v in enumerate(items) if type(v) in (int, float)}
        result['metadata_status'] = 'filename_public'
        if not header:
            return result
        url = 'https://drive.google.com/uc?export=download&id=' + file_id
        for _ in range(3):
            request = urllib.request.Request(url, headers={'Range': 'bytes=0-31', 'User-Agent': 'Mozilla/5.0'})
            with opener.open(request, timeout=40) as response:
                typ = response.headers.get('Content-Type', '')
                if 'text/html' not in typ:
                    prefix = response.read(32)
                    result.update(header_status='binary_header_read', binary_bytes_read=len(prefix),
                                  header_hex=prefix.hex(), status_code=response.status, content_type=typ,
                                  content_length=response.headers.get('Content-Length'),
                                  content_range=response.headers.get('Content-Range'),
                                  content_disposition=response.headers.get('Content-Disposition'))
                    return result
                page = response.read(262144).decode('utf8', 'replace')
            form = re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', page, re.S)
            if not form:
                result['header_status'] = 'html_without_public_download_form'
                return result
            action = html.unescape(form.group(1))
            host = urllib.parse.urlparse(action).hostname
            if host not in ('drive.google.com', 'drive.usercontent.google.com'):
                raise RuntimeError('Unexpected public download form host')
            fields = dict((html.unescape(k), html.unescape(v)) for k, v in re.findall(r'name="([^"]+)"[^>]*value="([^"]*)"', form.group(2)))
            if fields.get('id') != file_id:
                raise RuntimeError('Download form points at different file')
            url = action + '?' + urllib.parse.urlencode(fields)
        result['header_status'] = 'bounded_probe_exhausted'
    except Exception as e:
        result['error'] = type(e).__name__ + ': ' + str(e)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--read-header', action='store_true', help='Read at most32 binary bytes per file, never save weights')
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(lambda x: probe(x, args.read_header), FILES):
            print(json.dumps(row, ensure_ascii=False), flush=True)

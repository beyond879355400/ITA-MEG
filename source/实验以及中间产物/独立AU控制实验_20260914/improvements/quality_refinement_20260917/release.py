from pathlib import Path
import importlib.util
Q=Path(__file__).resolve().parent;EXP=Q.parents[1];ROOT=EXP.parents[1]
p=EXP/'completion_20260915/run_release_with_receipt.py'
s=importlib.util.spec_from_file_location('release_engine',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.HERE=Q;m.ROOT=ROOT;m.main()

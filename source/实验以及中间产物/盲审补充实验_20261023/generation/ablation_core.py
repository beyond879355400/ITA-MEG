"""Two explicit intervention switches; no model imports or file mutations."""
import hashlib

ARMS = (
    {"name": "P1S1", "projection": True, "smoothing": True},
    {"name": "P0S1", "projection": False, "smoothing": True},
    {"name": "P1S0", "projection": True, "smoothing": False},
    {"name": "P0S0", "projection": False, "smoothing": False},
)
SEEDS = [101, 202, 303]
TIMESTEPS = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]


def need(value, message):
    if not value:
        raise ValueError(message)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def select_cases(cache):
    """All 24 nonzero requests, in the original frozen order, never by score."""
    cases = [c for c in cache["cases"] if any(c["target_au_vector_0_5"])]
    need(len(cases) == 24 and len({c["case_id"] for c in cases}) == 24, "Expected 24 unique nonzero cases")
    need(len({c["subject"] for c in cases}) == 8, "Expected 8 original subjects")
    need(all(not any(c["source_au_vector_0_5"]) for c in cases), "Original zero-AU source contract changed")
    return cases


def execution_grid(cases, smoke=False):
    selected = cases[:1] if smoke else cases
    seeds = SEEDS[:1] if smoke else SEEDS
    return [(case, seed, arm) for case in selected for seed in seeds for arm in ARMS]


def make_controller(v11, unet, source, edit, cross, torch, cleanup, smoothing):
    class Controller(v11.v10.TerminalController):
        def __init__(self):
            super().__init__(unet, "terminal_last_eps", source, edit, cross, torch)

        def correct(self, latent, pipeline, prompt_embeds):
            clean = super().correct(latent, pipeline, prompt_embeds)
            result = v11.local_filtered_clean(clean, source, edit, cleanup, .5, torch) if smoothing else clean
            self.terminal_trace.update(
                mode="local_lp_s050" if smoothing else "terminal_last_eps_no_local_smoothing",
                base_mode="terminal_last_eps", smoothing_enabled=smoothing,
                base_terminal_clean_latent_sha256=tensor_sha(clean),
                post_filter_clean_latent_sha256=tensor_sha(result),
                filter=v11.filter_metadata(.5) if smoothing else None,
                outside_cleanup_clean_exact=True,
            )
            return result
    return Controller()


def sampling_callback(controller, source, noise, edit, torch, projection):
    history = []

    def step_end(pipeline, index, timestep, kwargs):
        latents = kwargs["latents"]
        need(torch.isfinite(latents).all().item(), "Nonfinite denoising latent")
        times = pipeline.scheduler.timesteps[5:]
        need([int(v) for v in times] == TIMESTEPS and int(timestep) == int(times[index]), "Frozen timesteps changed")
        next_t = times[index + 1] if index + 1 < len(times) else None
        if next_t is None:
            latents = controller.correct(latents, pipeline, kwargs["prompt_embeds"])
        if projection:
            reference = source if next_t is None else pipeline.scheduler.add_noise(source, noise, next_t.reshape(1))
            result = edit * latents + (1 - edit) * reference
        else:
            # Do not change either mask, source initialization, or terminal correction.
            result = latents
        need(torch.isfinite(result).all().item(), "Nonfinite final step latent")
        if next_t is None:
            controller.finalize_projection(result)
            controller.terminal_trace.update(projection_enabled=projection, final_latent_sha256=tensor_sha(result))
            if controller.terminal_trace.get('filter') is not None and not projection:
                controller.terminal_trace['filter']['final_projection'] = 'disabled'
        history.append({"t": int(timestep), "next_t": None if next_t is None else int(next_t),
                        "projection_applied": projection, "latent_sha256": tensor_sha(result)})
        return {"latents": result}
    return step_end, history

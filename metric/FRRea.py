import os


def compute_FRRea(result_dir, device="cpu", batch_size=32, dims=2048):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise ImportError(
            "FRRea requires `pytorch-fid`. Install it with `pip install pytorch-fid`."
        ) from exc

    real_dir = os.path.join(result_dir, "fid", "real")
    fake_dir = os.path.join(result_dir, "fid", "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        raise FileNotFoundError(
            "Missing fid image folders. Run evaluation with --compute_frrea first."
        )

    return calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
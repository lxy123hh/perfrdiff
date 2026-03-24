import os
import shutil
import argparse
import time
from copy import deepcopy

import torch
from tqdm import tqdm

from metric import *
from dataset.dataset import get_dataloader
from utils.util import load_config, init_seed
import model as module_arch

PROTOCOL_CHOICES = ("mafrg", "pmafrg", "both")
PROTOCOL_LABELS = {
    "mafrg": "MAFRG",
    "pmafrg": "PMAFRG",
}


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--config", type=str, help="config path", default="./configs/rewrite_weight.yaml")
    parser.add_argument(
        "--protocol",
        type=str,
        choices=PROTOCOL_CHOICES,
        default="pmafrg",
        help="evaluation protocol to run: mafrg, pmafrg, or both",
    )
    parser.add_argument("--compute_frrea", action="store_true")
    parser.add_argument("--frrea_dir", type=str, default="./results/rewrite_weight/frrea_eval")
    parser.add_argument("--frrea_period", type=int, default=10)
    parser.add_argument("--frrea_step", type=int, default=1)
    parser.add_argument("--frrea_fake_repeat", type=int, default=10)
    args = parser.parse_args()
    return args


def build_test_conf(base_conf, use_person_specific_split, compute_frrea=False):
    test_conf = deepcopy(base_conf)
    test_conf.use_person_specific_split = use_person_specific_split

    if compute_frrea:
        test_conf.load_video_s = False
        test_conf.load_video_l = True
        test_conf.load_ref = True
    else:
        test_conf.load_video_s = False
        test_conf.load_video_l = False
        test_conf.load_ref = False

    return test_conf


def collect_predictions(cfg, device, model, data_loader, compute_frrea=False, frrea_dir=None, frrea_period=10, frrea_step=1, frrea_fake_repeat=10):
    render = None
    if compute_frrea:
        from utils.render import Render

        fid_dir = os.path.join(frrea_dir, "fid")
        if os.path.exists(fid_dir):
            shutil.rmtree(fid_dir)
        os.makedirs(frrea_dir, exist_ok=True)
        render = Render("cuda" if device.type == "cuda" else "cpu")

    if frrea_period <= 0:
        raise ValueError("`frrea_period` must be a positive integer.")
    if frrea_step <= 0:
        raise ValueError("`frrea_step` must be a positive integer.")
    if frrea_fake_repeat <= 0:
        raise ValueError("`frrea_fake_repeat` must be a positive integer.")

    model.eval()

    speaker_emotion_list = []
    listener_emotion_gt_list = []
    listener_emotion_pred_list = []

    for batch_idx, (
        speaker_audio_clip,
        speaker_video_clip,
        speaker_emotion_clip,
        speaker_3dmm_clip,
        listener_video_clip,
        listener_emotion_clip,
        listener_3dmm_clip,
        listener_3dmm_clip_personal,
        listener_reference,
    ) in enumerate(tqdm(data_loader)):
        (
            speaker_audio_clip,
            speaker_video_clip,
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
        ) = (
            speaker_audio_clip.to(device),
            speaker_video_clip.to(device),
            speaker_emotion_clip.to(device),
            speaker_3dmm_clip.to(device),
            listener_video_clip.to(device),
            listener_emotion_clip.to(device),
            listener_3dmm_clip.to(device),
            listener_3dmm_clip_personal.to(device),
            listener_reference.to(device),
        )

        listener_emotion_gt = listener_emotion_clip.detach().clone().cpu()
        listener_emotion_clip = listener_emotion_clip.repeat_interleave(cfg.k_appro, dim=0)
        listener_3dmm_clip = listener_3dmm_clip.repeat_interleave(cfg.k_appro, dim=0)

        input_dict = {
            "speaker_audio": speaker_audio_clip,
            "speaker_emotion_input": speaker_emotion_clip,
            "speaker_3dmm_input": speaker_3dmm_clip,
            "listener_emotion_input": listener_emotion_clip,
            "listener_3dmm_input": listener_3dmm_clip,
            "listener_personal_input": listener_3dmm_clip_personal,
        }

        with torch.no_grad():
            b, l, d = speaker_emotion_clip.shape
            listener_emotion_preds = torch.zeros(
                size=(b, 10, l, d),
                device=device,
                dtype=speaker_emotion_clip.dtype,
            )

            for i in range(10):
                [_, listener_emotion_pred], _ = model(x=input_dict, p=listener_3dmm_clip_personal[:1])
                listener_emotion_preds[:, i:(i + 1), :, :] = listener_emotion_pred["prediction_emotion"]

            if compute_frrea and (batch_idx % frrea_period == 0):
                max_repeat = min(frrea_fake_repeat, listener_emotion_preds.shape[1])
                for fake_idx in range(max_repeat):
                    pred_3dmm = model.main_net.diffusion_decoder.latent_embedder.decode_coeff(
                        listener_emotion_preds[:, fake_idx]
                    )
                    render.rendering_for_fid(
                        frrea_dir,
                        f"{batch_idx:05d}_k{fake_idx:02d}",
                        pred_3dmm[0],
                        speaker_video_clip[0] if speaker_video_clip.shape[0] > 0 else None,
                        listener_reference[0],
                        listener_video_clip[0],
                        step=frrea_step,
                        save_real=fake_idx == 0,
                    )

            speaker_emotion_list.append(speaker_emotion_clip.detach().cpu())
            listener_emotion_pred_list.append(listener_emotion_preds.detach().cpu())
            listener_emotion_gt_list.append(listener_emotion_gt.detach().cpu())

    all_speaker_emotion = torch.cat(speaker_emotion_list, dim=0)
    all_listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim=0)
    all_listener_emotion_gt = torch.cat(listener_emotion_gt_list, dim=0)
    return all_speaker_emotion, all_listener_emotion_pred, all_listener_emotion_gt


def _fmt_metric(value):
    return f"{value:.5f}"


def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"


def _get_protocols(protocol):
    if protocol == "both":
        return ["mafrg", "pmafrg"]
    return [protocol]


def _get_frrea_dir(base_dir, selected_protocol, current_protocol):
    if selected_protocol == "both":
        return os.path.join(base_dir, current_protocol)
    return base_dir


def _evaluate_protocol(
    base_conf,
    device,
    model,
    protocol,
    compute_frrea=False,
    frrea_dir=None,
    frrea_period=10,
    frrea_step=1,
    frrea_fake_repeat=10,
):
    protocol_start_time = time.perf_counter()
    use_person_specific_split = protocol == "pmafrg"
    protocol_label = PROTOCOL_LABELS[protocol]
    protocol_desc = "person-specific" if use_person_specific_split else "generic"

    print(f"==> Evaluating {protocol_desc} test protocol for {protocol_label} metrics...")

    test_conf = build_test_conf(
        base_conf,
        use_person_specific_split=use_person_specific_split,
        compute_frrea=compute_frrea,
    )
    data_loader = get_dataloader(test_conf)
    prediction_start_time = time.perf_counter()
    speaker, pred, gt = collect_predictions(
        test_conf,
        device,
        model,
        data_loader,
        compute_frrea=compute_frrea,
        frrea_dir=frrea_dir,
        frrea_period=frrea_period,
        frrea_step=frrea_step,
        frrea_fake_repeat=frrea_fake_repeat,
    )
    prediction_elapsed = time.perf_counter() - prediction_start_time

    p = test_conf.threads
    metrics_start_time = time.perf_counter()
    metrics = {
        "frc": compute_FRC_mp(test_conf, pred, gt, val_test="test", p=p),
        "frd": compute_FRD_mp(test_conf, pred, gt, val_test="test", p=p),
        "frdvs": compute_FRDvs(pred).item() * 100,
        "frvar": compute_FRVar(pred) * 100,
        "frdiv": compute_s_mse(pred).item() * 100,
        "frsyn": compute_TLCC_mp(pred, speaker, p=p),
        "frrea": None,
    }
    metrics_elapsed = time.perf_counter() - metrics_start_time
    frrea_elapsed = 0.0

    if compute_frrea:
        frrea_start_time = time.perf_counter()
        metrics["frrea"] = compute_FRRea(
            frrea_dir,
            device="cuda" if device.type == "cuda" else "cpu",
        )
        frrea_elapsed = time.perf_counter() - frrea_start_time

    metrics["timing"] = {
        "prediction": prediction_elapsed,
        "metrics": metrics_elapsed,
        "frrea": frrea_elapsed,
        "total": time.perf_counter() - protocol_start_time,
    }

    return metrics


def _print_protocol_metrics(protocol, metrics, show_protocol_prefix_for_frrea=False):
    protocol_label = PROTOCOL_LABELS[protocol]
    print(
        f"{protocol_label}-FRCorr: {_fmt_metric(metrics['frc'])}  "
        f"{protocol_label}-FRdist: {_fmt_metric(metrics['frd'])}  "
        f"FRDiv(table): {_fmt_metric(metrics['frdiv'])}  "
        f"FRDvs(table): {_fmt_metric(metrics['frdvs'])}  "
        f"FRVar(table): {_fmt_metric(metrics['frvar'])}  "
        f"FRSyn: {_fmt_metric(metrics['frsyn'])}"
    )

    if metrics["frrea"] is not None:
        frrea_label = "FRRea(FID)"
        if show_protocol_prefix_for_frrea:
            frrea_label = f"{protocol_label}-{frrea_label}"
        print(f"{frrea_label}: {_fmt_metric(metrics['frrea'])}")

    timing = metrics.get("timing")
    if timing is not None:
        print(
            f"{protocol_label}-Time: total {_fmt_duration(timing['total'])}  "
            f"prediction {_fmt_duration(timing['prediction'])}  "
            f"metrics {_fmt_duration(timing['metrics'])}"
            + (
                f"  FRRea {_fmt_duration(timing['frrea'])}"
                if metrics["frrea"] is not None
                else ""
            )
        )


def main(args):
    total_start_time = time.perf_counter()
    cfg = load_config(args=args, config_path=args.config)
    init_seed(seed=cfg.trainer.seed)

    if torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    diff_model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    diff_model.to(device)

    main_model = getattr(module_arch, cfg.main_model.type)(cfg, diff_model, device)
    main_model.to(device)

    results = []
    for protocol in _get_protocols(args.protocol):
        protocol_frrea_dir = None
        if args.compute_frrea:
            protocol_frrea_dir = _get_frrea_dir(args.frrea_dir, args.protocol, protocol)

        metrics = _evaluate_protocol(
            cfg.test_dataset,
            device,
            main_model,
            protocol,
            compute_frrea=args.compute_frrea,
            frrea_dir=protocol_frrea_dir,
            frrea_period=args.frrea_period,
            frrea_step=args.frrea_step,
            frrea_fake_repeat=args.frrea_fake_repeat,
        )
        results.append((protocol, metrics))

    for protocol, metrics in results:
        _print_protocol_metrics(
            protocol,
            metrics,
            show_protocol_prefix_for_frrea=args.protocol == "both",
        )

    print(f"Total elapsed time: {_fmt_duration(time.perf_counter() - total_start_time)}")


if __name__ == "__main__":
    main(args=parse_arg())

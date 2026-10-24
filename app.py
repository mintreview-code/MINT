#!/usr/bin/env python3
"""Local Gradio controls for the weighted-score Linux/GPU campaign."""

import argparse
import contextlib
import io
from pathlib import Path
import shlex

import gradio as gr

import lf
from prepare import EXAMPLE, HERE, prepare_run


def prepare_from_ui(site, out, pdb, chain, fixed, direction, seed, mode, smoke):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        run = prepare_run(
            out,
            site,
            pdb=pdb or EXAMPLE,
            chain=chain,
            fixed=fixed,
            direction=direction,
            seed=int(seed) if seed is not None else None,
            native_mode=mode,
            smoke=smoke,
        )
    return (
        output.getvalue() + "\nNo job submitted.",
        str(run),
        lf.read(run / "config.json"),
    )


def inspect_run(path):
    run = Path(path).expanduser().resolve()
    result = {}
    for name in (
        "config.json",
        "submission.json",
        "live.json",
        "paused.json",
        "complete.json",
        "failure.json",
    ):
        file = run / name
        if file.exists():
            result[name] = lf.read(file)
    if not result:
        raise gr.Error("No run records found at this path.")
    return result


def preview_submission(path):
    return shlex.join(lf.submission_command(Path(path).expanduser().resolve()))


def submit_run(path, resume):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        lf.submit(Path(path).expanduser(), resume=resume)
    return output.getvalue()


def build_app():
    with gr.Blocks(title="Fast Search + AF3") as app:
        gr.Markdown(
            "# Fast Search + AF3\n"
            "Prepare a 120-round weighted-score search at joint temperature 0.4, submit it to Slurm, and inspect its progress. "
            "Full inference needs a configured Linux machine with one NVIDIA GPU, NetMHCpan, "
            "and the compatible AF3 container and parameters. The Mac can display these controls, "
            "but cannot run this AF3 backend.\n\n"
            "The bundled LuxSit seed and its 43 fixed positions are used when no PDB is uploaded. "
            "For a custom PDB, specify its own fixed residues."
        )
        with gr.Tab("Prepare"):
            site = gr.Textbox(
                label="Site configuration file on this machine",
                value=str(HERE / "configs/digs.example.json"),
            )
            out = gr.Textbox(
                label="New run directory", value=str(HERE / "runs/luxsit_deimmunize")
            )
            pdb = gr.File(
                label="Custom PDB (optional; otherwise use LuxSit)",
                file_types=[".pdb"],
                type="filepath",
            )
            chain = gr.Textbox(label="Chain (optional for a single-chain input)")
            fixed = gr.Textbox(
                label="Fixed residues for a custom PDB", placeholder="B:40,B:45A"
            )
            direction = gr.Radio(
                ["deimmunize", "immunize"], value="deimmunize", label="Direction"
            )
            seed = gr.Number(
                value=None,
                precision=0,
                label="Random seed (optional)",
                info="Blank uses 26051011 for the campaign or 202610071 for the small test.",
            )
            mode = gr.Dropdown(
                ["required", "auto", "off"],
                value="required",
                label="Original NetMHCpan accelerator",
                info="Campaign default: required. Use off for stock prediction on a supported Linux CPU without AVX2.",
            )
            smoke = gr.Checkbox(
                label="Small real-model test (includes real AF3 folds)", value=False
            )
            prepare_button = gr.Button("Prepare run")
            prepare_output = gr.Textbox(label="Preparation result")
            config = gr.JSON(label="Prepared configuration")
        with gr.Tab("Run and inspect"):
            run_path = gr.Textbox(label="Prepared run directory")
            gr.Markdown(
                "Preparing and previewing do not launch a job. **Submit to Slurm** launches one GPU job using the saved configuration. For a Linux GPU workstation without Slurm, use the command in the README."
            )
            preview_button = gr.Button("Preview Slurm command")
            resume = gr.Checkbox(label="Resume this run's own checkpoint", value=False)
            submit_button = gr.Button("Submit to Slurm")
            submission = gr.Textbox(label="Submission")
            refresh_button = gr.Button("Refresh progress")
            progress = gr.JSON(label="Saved run records")
        prepare_button.click(
            prepare_from_ui,
            [site, out, pdb, chain, fixed, direction, seed, mode, smoke],
            [prepare_output, run_path, config],
        )
        preview_button.click(preview_submission, [run_path], [submission])
        submit_button.click(submit_run, [run_path, resume], [submission])
        refresh_button.click(inspect_run, [run_path], [progress])
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )

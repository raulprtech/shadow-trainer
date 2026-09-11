"""Command-line interface for the Shadow Trainer MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import JobConfig
from .demo import run_demo
from .errors import ShadowTrainerError
from .reporting import render_report
from .resources import print_snapshot, snapshot
from .runtime import inspect_job, resume_job, run_job


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shadow-trainer",
        description="Auditable training under bounded VRAM, RAM, disk, and data locality.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="inspect host resources and dependencies")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--disk-path", type=Path, default=Path("/"))
    plan = commands.add_parser("plan", help="validate and admit a job without running it")
    plan.add_argument("job_config", type=Path)
    plan.add_argument("--json", action="store_true")
    run = commands.add_parser("run", help="execute a new job")
    run.add_argument("job_config", type=Path)
    resume = commands.add_parser("resume", help="resume a durable run")
    resume.add_argument("run_dir", type=Path)
    report = commands.add_parser("report", help="regenerate a run report")
    report.add_argument("run_dir", type=Path)
    demo = commands.add_parser("demo", help="run the portable local 3D-CNN demo")
    demo.add_argument("--output-dir", required=True, type=Path)
    demo.add_argument("--cpu", action="store_true", help="allow a CPU-only demonstration")
    return parser


def _print_error(exc: Exception) -> None:
    if isinstance(exc, ShadowTrainerError):
        payload = {"error": exc.code, "message": str(exc), "details": exc.details}
    else:
        payload = {"error": type(exc).__name__, "message": str(exc)}
    print(json.dumps(payload, indent=2), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            print_snapshot(snapshot(args.disk_path), args.json)
            return 0
        if args.command == "plan":
            config = JobConfig.load(args.job_config)
            environment, plan = inspect_job(config)
            payload = {"plan": plan.as_dict(), "environment": environment}
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("ADMITTED" if plan.admitted else "REJECTED")
                print(f"Strategy: {plan.selected_strategy}")
                for warning in plan.warnings:
                    print(f"Warning: {warning}")
                for reason in plan.reasons:
                    print(f"Reason: {reason}")
            return 0 if plan.admitted else 2
        if args.command == "run":
            result = run_job(JobConfig.load(args.job_config))
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "success" else 3
        if args.command == "resume":
            result = resume_job(args.run_dir)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "success" else 3
        if args.command == "report":
            print(render_report(args.run_dir))
            return 0
        if args.command == "demo":
            result = run_demo(args.output_dir, use_cuda=not args.cpu)
            print(json.dumps(result, indent=2, sort_keys=True))
            print(f"Report: {(args.output_dir / 'run' / 'report.html').resolve()}")
            return 0
    except (ShadowTrainerError, OSError, ValueError, RuntimeError) as exc:
        _print_error(exc)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

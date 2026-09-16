#!/usr/bin/env python3
"""Eight-hour fail-closed STU-Net campaign supervisor."""
from __future__ import annotations
import argparse,csv,fcntl,json,os,shutil,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
SHADOW=HERE.parent
LAB=Path("/home/raulprtech/stream-hot-kits-mini")
sys.path[:0]=[str(HERE),str(LAB)]
from stage23_bounded_nifti import BoundedWindowStager
from stage34_resilient_staging import ProgressAwareRcloneSource
from stunet_campaign import atomic_json,csv_case_ids,manifest_records,prior_used_cases,record_bytes,select_evaluation_cases,sha256_file
GIB=2**30;MIB=2**20

def snapshot(pid=None):
    memory=psutil.virtual_memory();swap=psutil.swap_memory();rss=0
    if pid:
        try:
            parent=psutil.Process(pid)
            for process in [parent,*parent.children(recursive=True)]:
                try:rss+=process.memory_info().rss
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass
    return {"time":time.time(),"disk_free_bytes":shutil.disk_usage("/mnt/c").free,
            "available_ram_bytes":memory.available,"swap_used_bytes":swap.used,
            "process_tree_rss_bytes":rss}

def failures(state,startup=False,swap_limit=256*MIB):
    result=[]
    if state["disk_free_bytes"]<20*GIB+(5*GIB if startup else 0):result.append("physical_disk_floor")
    if state["available_ram_bytes"]<(5*GIB if startup else 512*MIB):result.append("available_ram_floor")
    if state["swap_used_bytes"]>swap_limit:result.append("swap_limit")
    if state["process_tree_rss_bytes"]>int(4.5*GIB):result.append("process_tree_rss_limit")
    return result

def stop_group(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)

def load_json(path):return json.loads(Path(path).read_text())

def directory_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())

def preflight(config,require_remote=True):
    state=snapshot();issues=failures(state,startup=True)
    paths=[Path(config[key]) for key in ("manifest","base_checkpoint","train_csv","validation_csv","cache_seed","stage34_schedule")]
    for path in paths:
        if not path.exists():issues.append(f"missing:{path}")
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free","--format=csv,noheader,nounits"],
                       capture_output=True,text=True,timeout=20)
    remote={"checked":require_remote,"ok":None,"stderr":""}
    if require_remote:
        probe=subprocess.run(["rclone","lsf",f"{config['remote']}/case_00569","--max-depth","1",
            "--contimeout","10s","--timeout","20s","--retries","1","--low-level-retries","1"],
            capture_output=True,text=True,timeout=35)
        remote={"checked":True,"ok":probe.returncode==0,"stdout":probe.stdout.strip(),"stderr":probe.stderr.strip()}
        if probe.returncode:issues.append("remote_unavailable")
    if paths[1].exists() and sha256_file(paths[1])!=config["base_checkpoint_sha256"]:
        issues.append("base_checkpoint_digest")
    return {"schema_version":"shadowtrainer.stunet-preflight/v1","status":"pass" if not issues else "fail",
            "issues":issues,"resources":state,"gpu":{"returncode":gpu.returncode,"stdout":gpu.stdout.strip(),
            "stderr":gpu.stderr.strip()},"remote":remote}

def build_cohort(config):
    records=manifest_records(Path(config["manifest"]));frozen=load_json(config["stage34_schedule"])
    train=list(frozen["train_cases"]);development=list(frozen["validation_cases"])
    prior=prior_used_cases(LAB/"runs");candidates=csv_case_ids(Path(config["validation_csv"]))
    ranked=select_evaluation_cases(candidates,prior,seed=config["seed"],count=6)
    maximum=int(config["evaluation_cache_bytes"]);evaluation=[];occupancy=0
    for case_id in ranked:
        size=record_bytes(records[case_id])
        if occupancy+size<=maximum:evaluation.append(case_id);occupancy+=size
    if len(evaluation)<4:raise RuntimeError("fewer than four independent cases fit evaluation cache")
    overlap={"train_development":sorted(set(train)&set(development)),
             "train_evaluation":sorted(set(train)&set(evaluation)),
             "development_evaluation":sorted(set(development)&set(evaluation))}
    if any(overlap.values()):raise RuntimeError(f"cohort leakage: {overlap}")
    files=[]
    for case_id in [*train,*development,*evaluation]:
        for kind in ("image","label"):
            row=records[case_id][kind]
            files.append({"case_id":case_id,"kind":kind,"source":row["source"],
                          "size_bytes":row["size_bytes"],"md5":row.get("md5")})
    return {"schema_version":"shadowtrainer.stunet-cohort/v1","seed":config["seed"],
            "train_cases":train,"development_cases":development,"evaluation_cases":evaluation,
            "evaluation_ranked_candidates":ranked,"evaluation_cache_bytes":occupancy,
            "overlap":overlap,"files":files,"selection":"sha256(seed:case_id), ascending",
            "independent_evaluation":len(evaluation)>=4}

def stage_evaluation(config,cohort,events):
    raw=load_json(config["manifest"])
    sizes={row[kind]["source"]:row[kind]["size_bytes"] for row in raw["cases"] for kind in ("image","label")}
    source=ProgressAwareRcloneSource(config["remote"],sizes,"/mnt/c",20*GIB,attempts=3,
        stall_seconds=180,attempt_seconds=1800,poll_seconds=2)
    stager=BoundedWindowStager(Path(config["manifest"]),Path(config["evaluation_cache"]),
        int(config["evaluation_cache_bytes"]),source,validate_nifti=True)
    started=time.time();stager.stage_window(cohort["evaluation_cases"])
    result={"status":"success","cases":cohort["evaluation_cases"],"seconds":time.time()-started,
            "cache":stager.status(),"transfers":source.transfers,"attempts":source.attempt_log}
    with Path(events).open("a") as stream:stream.write(json.dumps({"kind":"evaluation_staged",**result})+"\n")
    return result

def run_guarded(command,cwd,log,resources,deadline,phase_deadline,swap_limit=256*MIB):
    env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    started=time.time();status="running";reasons=[];sample_index=0;session=Path(log).parent
    with Path(log).open("a") as output,Path(resources).open("a") as samples:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                state=snapshot(process.pid);samples.write(json.dumps(state)+"\n");samples.flush()
                reasons=failures(state,swap_limit=swap_limit);sample_index+=1
                if sample_index%10==0 and directory_bytes(session)>5*GIB:reasons.append("campaign_artifact_ceiling")
                if time.time()>=min(deadline,phase_deadline):reasons.append("time_budget")
                if reasons:status="guard_stopped";stop_group(process);break
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
            if status=="running":status="success" if process.returncode==0 else "failed"
        finally:stop_group(process)
    return {"status":status,"returncode":process.returncode,"reasons":reasons,
            "seconds":time.time()-started,"command":command,"log":str(log),"resources":str(resources)}

def training_step_median(metrics):
    stamps=[]
    for line in Path(metrics).read_text().splitlines():
        row=json.loads(line)
        if row.get("kind")=="train_step":stamps.append(row.get("time",row.get("wall_time")))
    stamps=[value for value in stamps if value is not None]
    deltas=[right-left for left,right in zip(stamps,stamps[1:]) if right>=left]
    return statistics.median(deltas) if deltas else 3.0

def phase_succeeded(state, name):
    return state.get("phases", {}).get(name, {}).get("status") == "success"

def historical_active_seconds(state):
    """Best-effort migration for sessions created before active-time accounting."""
    if "active_seconds" in state:
        return float(state["active_seconds"])
    return sum(
        float(phase.get("seconds", 0.0))
        for phase in state.get("phases", {}).values()
        if isinstance(phase, dict)
    )

def report(session):
    rows=[]
    for cohort in ("development","evaluation"):
        for model in ("B0","A","B"):
            path=session/"evaluation"/cohort/model/"summary.json"
            if not path.exists():continue
            payload=load_json(path)
            if payload.get("status")!="success":continue
            aggregate=payload["aggregate"]
            rows.append({"cohort":cohort,"model":model,
                "kidney":aggregate["classes"]["1"]["mean_dice"],
                "tumor":aggregate["classes"]["2"]["mean_dice"],
                "cyst":aggregate["classes"]["3"]["mean_dice"],
                "kidney_and_masses":aggregate["hec"]["kidney_and_masses"]["mean_dice"],
                "kidney_mass":aggregate["hec"]["kidney_mass"]["mean_dice"],
                "tumor_hec":aggregate["hec"]["tumor"]["mean_dice"]})
    csv_path=session/"results.csv"
    if rows:
        with csv_path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    evaluation={row["model"]:row for row in rows if row["cohort"]=="evaluation"};signal=False;best=None
    if {"B0","A","B"}<=set(evaluation):
        best=max((evaluation["A"],evaluation["B"]),key=lambda row:row["tumor"])
        signal=best["tumor"]-evaluation["B0"]["tumor"]>=.03 and best["kidney"]>=evaluation["B0"]["kidney"]-.02
    html_rows="".join("<tr>"+"".join(f"<td>{row[key]:.4f}</td>" if isinstance(row[key],float)
        else f"<td>{row[key]}</td>" for key in row)+"</tr>" for row in rows)
    report_path=session/"report.html"
    report_path.write_text("""<!doctype html><meta charset="utf-8"><title>STU-Net campaign</title>
<style>body{font-family:system-ui;max-width:1100px;margin:40px auto;color:#17202a}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd;padding:8px}th{background:#eef}</style>
<h1>STU-Net · campaña física RTX 3050 Ti</h1>
<p>Comparación B0, pérdida Stage20 y pérdida jerárquica renal/tumor.</p>
<table><thead><tr><th>Cohorte</th><th>Modelo</th><th>Riñón</th><th>Tumor</th><th>Quiste</th>
<th>Riñón+masas</th><th>Masa renal</th><th>Tumor HEC</th></tr></thead><tbody>"""+html_rows+
        "</tbody></table><h2>Lectura</h2><p>Señal preliminar: <strong>"+
        ("sí" if signal else "no")+"</strong>. Las métricas HEC siguen KiTS23 y no son puntuaciones oficiales.</p>")
    slides=session/"presentation.md"
    slides.write_text("# STU-Net en 4 GiB\n\n## Problema\nEntrenamiento 3D reproducible con recursos limitados.\n\n"
        "## Protocolo\nB0 vs Stage20 vs pérdida jerárquica; pacientes separados.\n\n"
        "## Recursos\nRTX 3050 Ti, staging por casos, checkpoints y guardas.\n\n"
        "## Resultados\nConsultar results.csv y overlays por paciente.\n\n"
        f"## Señal preliminar\n{'Sí' if signal else 'No'} según el gate predefinido.\n\n"
        "## Circuito14\nRuntime auditable para entrenamiento local limitado.\n\n"
        "## Siguiente\nRéplicas, nnU-Net y validación externa.\n")
    result={"status":"success" if len(rows)==6 else "partial","rows":rows,"promising_signal":signal,"best_candidate":best,
            "report":str(report_path),"slides":str(slides),"csv":str(csv_path)}
    atomic_json(session/"campaign-results.json",result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path)
    parser.add_argument("--preflight-only",action="store_true");parser.add_argument("--max-hours",type=float,default=8)
    parser.add_argument("--resume",type=Path);args=parser.parse_args()
    invocation_started=time.time()
    if args.resume:session=args.resume.resolve();config_path=session/"resolved-config.json"
    else:
        if not args.config:parser.error("--config required unless --resume is used")
        config_path=args.config.resolve();base=load_json(config_path)
        session=(Path(base["output_root"])/base["session_name"]).resolve()
    config=load_json(config_path);check=preflight(config,True)
    if not args.preflight_only and check["issues"]==["available_ram_floor"]:
        wait_deadline=time.time()+1800
        while time.time()<wait_deadline and check["issues"]==["available_ram_floor"]:
            time.sleep(15);check=preflight(config,True)
    if args.preflight_only:
        print(json.dumps(check,indent=2));return 0 if check["status"]=="pass" else 2
    if check["status"]!="pass":raise RuntimeError("preflight failed: "+",".join(check["issues"]))
    lock=(LAB/".stage23_supervisor.lock").open("a");fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume:
        if session.exists():raise RuntimeError(f"refusing to overwrite session: {session}")
        session.mkdir(parents=True);config=dict(config)
        config["training_cache"]=str(session/"training_cache")
        config["evaluation_cache"]=str(session/"evaluation_cache")
        atomic_json(session/"resolved-config.json",config)
    state_path=session/"campaign-state.json";events=session/"events.jsonl"
    state=load_json(state_path) if state_path.exists() else {
        "schema_version":"shadowtrainer.stunet-campaign/v1","status":"running",
        "started_at":time.time(),"phases":{}}
    previous_active_seconds=historical_active_seconds(state)
    if state.get("status") != "running":
        state.setdefault("invocation_history", []).append({
            "status": state.get("status"), "finished_at": state.get("finished_at"),
            "error": state.get("error"),
        })
    state["status"]="running"
    state.pop("error",None);state.pop("traceback",None)
    state["active_seconds"]=previous_active_seconds
    deadline=invocation_started+max(0.0,args.max_hours*3600-previous_active_seconds)
    def active_deadline(hours):
        return invocation_started+max(0.0,hours*3600-previous_active_seconds)
    def save_state():
        state["updated_at"]=time.time();state["resources"]=snapshot();atomic_json(state_path,state)
    def store_attempt(name,result):
        previous=state["phases"].get(name)
        if previous and previous.get("status")!="success":
            state.setdefault("attempt_history",{}).setdefault(name,[]).append(previous)
        state["phases"][name]=result;save_state()
    try:
        if "cohort" not in state["phases"]:
            cohort=build_cohort(config);atomic_json(session/"cohort.json",cohort)
            state["phases"]["cohort"]={"status":"success","evaluation_cases":cohort["evaluation_cases"]};save_state()
        else:cohort=load_json(session/"cohort.json")
        if "cache_seed" not in state["phases"]:
            destination=Path(config["training_cache"]);temporary=destination.with_name(destination.name+".partial")
            if destination.exists() and (destination/"cache_state.json").is_file():
                pass
            elif temporary.exists():
                raise RuntimeError(f"partial cache seed requires inspection: {temporary}")
            else:
                shutil.copytree(Path(config["cache_seed"]),temporary);os.replace(temporary,destination)
            state["phases"]["cache_seed"]={"status":"success"};save_state()
        if "evaluation_staging" not in state["phases"]:
            state["phases"]["evaluation_staging"]=stage_evaluation(config,cohort,events);save_state()
        python=config["python"]
        if not phase_succeeded(state,"pilot"):
            result=run_guarded([python,str(HERE/"stunet_campaign_worker.py"),"--config",
                str(session/"resolved-config.json"),"--session",str(session),"--arm","A","--epochs","1","--pilot"],
                LAB,session/"pilot.log",session/"pilot.resources.jsonl",deadline,
                min(deadline,active_deadline(1.5)))
            store_attempt("pilot",result)
            if result["status"]!="success":raise RuntimeError("pilot failed")
        median=training_step_median(session/"pilot_A"/"metrics.jsonl")
        projected=median*24*8*4*2*1.5
        epochs=4 if time.time()+projected+3*3600<deadline else 2
        state["epochs_selected"]=epochs;state["pilot_step_median_seconds"]=median;save_state()
        training_deadline=min(deadline,active_deadline(4.75))
        for arm in ("A","B"):
            phase=f"train_{arm}"
            if state["phases"].get(phase,{}).get("status")=="success":continue
            result=run_guarded([python,str(HERE/"stunet_campaign_worker.py"),"--config",
                str(session/"resolved-config.json"),"--session",str(session),"--arm",arm,"--epochs",str(epochs)],
                LAB,session/f"train_{arm}.log",session/f"train_{arm}.resources.jsonl",deadline,training_deadline)
            store_attempt(phase,result)
            if result["status"]!="success":break
        incomplete_training=[arm for arm in ("A","B") if not phase_succeeded(state,f"train_{arm}")]
        if incomplete_training:
            raise RuntimeError(f"training incomplete: {incomplete_training}")
        evaluation_deadline=min(deadline,active_deadline(7))
        for cohort_name,development in (("development",True),("evaluation",False)):
            for model in ("B0","A","B"):
                if model!="B0" and not (session/f"arm_{model}"/"best.checkpoint.pt").exists():continue
                phase=f"evaluate_{cohort_name}_{model}"
                if state["phases"].get(phase,{}).get("status")=="success":continue
                command=[python,str(HERE/"stunet_campaign_evaluate.py"),"--config",
                    str(session/"resolved-config.json"),"--session",str(session),"--model",model]
                if development:command.append("--development")
                result=run_guarded(command,LAB,session/f"{phase}.log",session/f"{phase}.resources.jsonl",
                    deadline,evaluation_deadline,192*MIB)
                store_attempt(phase,result)
                if result["status"]!="success":
                    raise RuntimeError(f"evaluation incomplete: {phase}")
        required_evaluations=[f"evaluate_{cohort}_{model}" for cohort in ("development","evaluation") for model in ("B0","A","B")]
        missing=[phase for phase in required_evaluations if not phase_succeeded(state,phase)]
        if missing:raise RuntimeError(f"evaluation incomplete: {missing}")
        state["results"]=report(session);state["status"]="success"
    except Exception as exc:
        state.update({"status":"error","error":repr(exc),"traceback":traceback.format_exc()})
        try:state["results"]=report(session)
        except Exception as report_error:state["report_error"]=repr(report_error)
    finally:
        state["finished_at"]=time.time()
        state["active_seconds"]=previous_active_seconds+(state["finished_at"]-invocation_started)
        state["duration_seconds"]=state["active_seconds"]
        state["wall_seconds"]=state["finished_at"]-state["started_at"]
        save_state()
    print(json.dumps(state,indent=2));return 0 if state["status"]=="success" else 1
if __name__=="__main__":raise SystemExit(main())

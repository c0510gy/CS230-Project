from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import json
import os
import subprocess, time
from threading import Thread
from . import background
import shlex
import psutil
# Create your views here.

proc = [None]
opt_thread = [None]
console_out_txt = []

def run_optimization_():
    
    cmds = 'source ~/.zshrc; source ./venv/bin/activate; cd SparkApp; spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 --repositories https://repos.spark-packages.org opt.py'
    # args = shlex.split(cmds)
    print('args', cmds)

    proc[0] = subprocess.Popen(cmds, stdout=subprocess.PIPE, cwd='/Users/sang-geon/git_reps/CS230-Project/', shell=True)


    # if proc[0] is not None:
    #   print('check')
    #   # print(proc[0].async_stdout.readlines())
    while proc[0] is not None:
        while True:
            line = proc[0].stdout.readline()
            if not line:
                break
            print(line.rstrip())
            console_out_txt.append(line.decode("utf-8"))
        time.sleep(1)


def run_optimization(request):
    if opt_thread[0] is None:
        opt_thread[0] = Thread(target=run_optimization_, ) #args = (10, )
        opt_thread[0].start()
    
    return HttpResponse(
        json.dumps({'status': 'success'}),
        content_type = 'application/javascript; charset=utf8'
    )

def stop_optimization(request):
    try:
      if proc[0] is not None:
          print(proc[0])
          process = psutil.Process(proc[0].pid)
          for proc_ in process.children(recursive=True):
              proc_.kill()
          process.kill()
          proc[0] = None
          opt_thread[0] = None

      return HttpResponse(
          json.dumps({'status': 'success'}),
          content_type = 'application/javascript; charset=utf8'
      ) 
    except Exception as e:
      print(e)

        
      return HttpResponse(
          json.dumps({'status': 'fail'}),
          content_type = 'application/javascript; charset=utf8'
      )

def get_optmization_status(request):
    
    opt_history_file = os.path.join(settings.BASE_DIR, '../SparkApp/opt_history.json')
    print(opt_history_file)
    opt_history = []
    if os.path.exists(opt_history_file):
        with open(opt_history_file, 'r') as f:
            opt_history = json.load(f)
    
    proc_running = not (proc[0] is None)
    
    dat = {
        'hello': 'world',
        'running': proc_running,
        'console_out': console_out_txt,
        'opt_history': opt_history,
    }

    return HttpResponse(
        json.dumps(dat),
        content_type = 'application/javascript; charset=utf8'
    )

def get_visualization(request):
    
    num = int(request.GET.get('num', 1))
    
    with open(os.path.join(settings.BASE_DIR, f'../SparkApp/visualization/opt_{num:02d}.png'), 'rb') as f:
        return HttpResponse(f.read(), content_type='image/jpeg')
    

#!/usr/bin/env python3
"""SSH transport: credentials enter via stdin and stay in memory only."""
import json
import os
import subprocess
import sys
import termios

if sys.stdin.isatty():
    attributes = termios.tcgetattr(sys.stdin.fileno())
    attributes[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON)
    attributes[6][termios.VMIN] = 1
    attributes[6][termios.VTIME] = 0
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attributes)
print(json.dumps({'awaiting_authentication': True, 'stdin_echo': False}), flush=True)
auth = json.loads(sys.stdin.readline())
env = os.environ.copy()
env['SSHPASS'] = auth['password']
address = auth['username'] + '@' + auth['host']
options = ['-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
print(json.dumps({'transport_ready': True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    action = request.get('action', 'run')
    if action == 'close':
        print(json.dumps({'closed': True}), flush=True)
        break
    if action == 'run':
        command = ['sshpass', '-e', 'ssh', *options, '-p', str(auth['port']),
                   address, request['command']]
    elif action == 'upload':
        command = ['sshpass', '-e', 'scp', *options, '-P', str(auth['port']),
                   request['local'], address + ':' + request['remote']]
    elif action == 'download':
        command = ['sshpass', '-e', 'scp', *options, '-P', str(auth['port']),
                   address + ':' + request['remote'], request['local']]
    else:
        raise ValueError('Unknown SSH action')
    try:
        input_options = {'input': request['stdin']} if 'stdin' in request else {'stdin': subprocess.DEVNULL}
        completed = subprocess.run(command, text=True, capture_output=True, env=env,
                                   timeout=request.get('timeout', 60), **input_options)
        result = {'id': request.get('id'), 'exit_code': completed.returncode,
                  'stdout': completed.stdout, 'stderr': completed.stderr}
    except subprocess.TimeoutExpired as error:
        result = {'id': request.get('id'), 'timeout': True,
                  'stdout': (error.stdout or b'').decode(errors='replace') if isinstance(error.stdout, bytes) else error.stdout,
                  'stderr': (error.stderr or b'').decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr}
    print(json.dumps(result), flush=True)

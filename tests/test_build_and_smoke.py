import os
import re
import shutil
import subprocess
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def binaries():
    if not shutil.which('mpicc') or not shutil.which('mpiexec'):
        pytest.skip('MPI development tools required')
    result=subprocess.run(['make','serial','mpi'],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    return ROOT

@pytest.mark.parametrize('ranks',[1,2])
def test_mpi_delivers_signals_and_terminates(binaries,tmp_path,ranks):
    result=subprocess.run(['mpiexec','-n',str(ranks),str(ROOT/'brain_mpi'),str(ROOT/'tests/tiny.graph'),'1'],cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    report=(tmp_path/'summary_report.generated').read_text()
    assert '1 neurons, 1 nerves' in report
    assert 'until 1 ns' in report
    match=re.search(r'ID: 1\), total signals received: (\d+)',report)
    assert match and int(match.group(1))>0,report
    assert 'Invalid' not in result.stderr

@pytest.mark.parametrize('duration',['-1','invalid'])
def test_bad_duration_fails_promptly(binaries,tmp_path,duration):
    result=subprocess.run(['mpiexec','-n','2',str(ROOT/'brain_mpi'),str(ROOT/'tests/tiny.graph'),duration],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode!=0
    assert not (tmp_path/'summary_report.generated').exists()

def test_bad_node_id_is_rejected(binaries,tmp_path):
    graph=tmp_path/'bad.graph'
    graph.write_text((ROOT/'tests/tiny.graph').read_text().replace('<id>0</id>','<id>-1</id>'))
    result=subprocess.run(['mpiexec','-n','2',str(ROOT/'brain_mpi'),str(graph),'1'],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode!=0

@pytest.mark.parametrize('capacity',['1e-20','nan'])
def test_invalid_capacity_does_not_hang(binaries,tmp_path,capacity):
    import signal
    graph=tmp_path/'capacity.graph'
    graph.write_text((ROOT/'tests/tiny.graph').read_text().replace('<max_value>1000</max_value>',f'<max_value>{capacity}</max_value>'))
    proc=subprocess.Popen(['mpiexec','-n','1',str(ROOT/'brain_mpi'),str(graph),'1'],cwd=tmp_path,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:
        stdout,stderr=proc.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.communicate()
        pytest.fail('Signal propagation hung on invalid edge capacity')
    assert proc.returncode!=0,stdout+stderr

def test_nonroot_nerve_statistics_are_gathered(binaries,tmp_path):
    source=(ROOT/'tests/tiny.graph').read_text()
    nerve=source[source.index('<nerve>'):source.index('</nerve>')+len('</nerve>')]
    neuron=source[source.index('<neuron>'):source.index('</neuron>')+len('</neuron>')]
    source=source.replace(nerve+'\n'+neuron,neuron+'\n'+nerve)
    graph=tmp_path/'nonroot.graph';graph.write_text(source)
    result=subprocess.run(['mpiexec','-n','2',str(ROOT/'brain_mpi'),str(graph),'1'],cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    report=(tmp_path/'summary_report.generated').read_text()
    assert any(int(n)>0 for n in re.findall(r'Type \d+: (\d+) inputs',report)),report


def test_serial_terminal_neuron_does_not_crash(binaries,tmp_path):
    result=subprocess.run([str(ROOT/'brain_serial'),str(ROOT/'tests/tiny.graph'),'1'],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    report=(tmp_path/'summary_report.generated').read_text()
    match=re.search(r'total signals received (\d+)',report)
    assert match and int(match.group(1))>0,report

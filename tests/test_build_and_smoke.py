import os
import re
import shutil
import subprocess
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def serial_binary():
    result=subprocess.run(['make','serial'],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    return ROOT

@pytest.fixture(scope='module')
def binaries(serial_binary):
    if not shutil.which('mpicc') or not shutil.which('mpiexec'):
        pytest.skip('MPI development tools required')
    result=subprocess.run(['make','mpi'],cwd=ROOT,capture_output=True,text=True,timeout=60)
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


def test_serial_terminal_neuron_does_not_crash(serial_binary,tmp_path):
    result=subprocess.run([str(ROOT/'brain_serial'),str(ROOT/'tests/tiny.graph'),'1'],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    report=(tmp_path/'summary_report.generated').read_text()
    match=re.search(r'total signals received (\d+)',report)
    assert match and int(match.group(1))>0,report


def run_serial(serial_binary, tmp_path, graph, duration='0'):
    return subprocess.run([str(serial_binary/'brain_serial'),str(graph),duration],cwd=tmp_path,capture_output=True,text=True,timeout=10)


@pytest.mark.parametrize('duration',['-1','invalid','1.5'])
def test_serial_rejects_invalid_duration(serial_binary,tmp_path,duration):
    result=run_serial(serial_binary,tmp_path,ROOT/'tests/tiny.graph',duration)
    assert result.returncode!=0
    assert not (tmp_path/'summary_report.generated').exists()


@pytest.mark.parametrize('old,new',[
    ('<id>0</id>','<id>1.5</id>'),
    ('<id>1</id>','<id>7</id>'),
    ('<type>sensory</type>',''),
    ('<to>1</to>','<to>7</to>'),
    ('<max_value>1000</max_value>','<max_value>0</max_value>'),
    ('<weighting_0>1</weighting_0>','<weighting_0>nan</weighting_0>'),
])
def test_serial_rejects_invalid_graph(serial_binary,tmp_path,old,new):
    graph=tmp_path/'bad.graph'
    graph.write_text((ROOT/'tests/tiny.graph').read_text().replace(old,new))
    result=run_serial(serial_binary,tmp_path,graph)
    assert result.returncode!=0,result.stdout+result.stderr


@pytest.mark.parametrize('old,new',[
    ('<id>0</id>',''),
    ('<id>0</id>','<id>abc</id>'),
    ('<type>sensory</type>',''),
    ('<to>1</to>','<to>99</to>'),
    ('<direction>unidirectional</direction>','<direction>sideways</direction>'),
    ('</edge>',''),
    ('</neuron>',''),
])
def test_mpi_rejects_malformed_graph(binaries,tmp_path,old,new):
    graph=tmp_path/'bad.graph'
    graph.write_text((ROOT/'tests/tiny.graph').read_text().replace(old,new))
    result=subprocess.run(['mpiexec','-n','2',str(ROOT/'brain_mpi'),str(graph),'0'],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode!=0,result.stdout+result.stderr


@pytest.mark.parametrize('graph_name',['small','medium','large','massive'])
def test_mpi_loads_bundled_graphs_at_duration_zero(binaries,tmp_path,graph_name):
    result=subprocess.run(['mpiexec','-n','1',str(ROOT/'brain_mpi'),str(ROOT/graph_name),'0'],cwd=tmp_path,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('old,new',[
    ('<type>sensory</type>','<type>sensory'),
    ('<direction>unidirectional</direction>','<direction>unidirectional'),
    ('<weighting_0>1</weighting_0>','<weighting_0>1'),
])
def test_serial_rejects_missing_string_delimiter_without_signal(serial_binary,tmp_path,old,new):
    graph=tmp_path/'bad.graph'
    graph.write_text((ROOT/'tests/tiny.graph').read_text().replace(old,new))
    result=run_serial(serial_binary,tmp_path,graph)
    assert result.returncode > 0,result.stdout+result.stderr
    assert 'Invalid graph:' in result.stderr


@pytest.mark.parametrize('old,new',[
    ('<type>sensory</type>','<type>sensoryJUNK</type>'),
    ('</neuron>','</neuron>\n<type>sensory</type>'),
])
def test_mpi_rejects_invalid_type_state_or_token(binaries,tmp_path,old,new):
    graph=tmp_path/'bad.graph'
    graph.write_text((ROOT/'tests/tiny.graph').read_text().replace(old,new))
    result=subprocess.run(['mpiexec','-n','2',str(ROOT/'brain_mpi'),str(graph),'0'],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode!=0,result.stdout+result.stderr
    assert 'Invalid graph:' in result.stderr


def duplicate_edge_graph(tmp_path):
    graph=tmp_path/'duplicate.graph'
    graph.write_text((ROOT/'tests/tiny.graph').read_text().replace('<to>1</to>','<to>0</to>\n<to>1</to>'))
    return graph


def test_serial_duplicate_edge_field_is_rejected(serial_binary,tmp_path):
    result=run_serial(serial_binary,tmp_path,duplicate_edge_graph(tmp_path))
    assert result.returncode!=0,result.stdout+result.stderr


def test_mpi_duplicate_edge_field_is_rejected(binaries,tmp_path):
    graph=duplicate_edge_graph(tmp_path)
    result=subprocess.run(['mpiexec','-n','2',str(ROOT/'brain_mpi'),str(graph),'0'],cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode!=0,result.stdout+result.stderr

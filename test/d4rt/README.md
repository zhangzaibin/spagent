# D4RT Mock Test

This folder contains the mock test for the D4RT tool integration in SPAgent.

## What it does

This test verifies that:

- `D4RTTool` can be imported from `spagent.tools`
- the local D4RT mock server is running
- the tool can send a request to the server
- the server returns the expected mock result
- mock output files can be generated successfully

## What it does not do

This test does **not** run the real D4RT model.  
It only checks whether the integration pipeline is working.

## Required files

This test depends on:

- `spagent/tools/d4rt_tool.py`
- `spagent/tools/__init__.py`
- `external_experts/D4RT/d4rt_server.py`

## Start the mock server

```bash
cd external_experts/D4RT
python d4rt_server.py
```



Then check:

```
curl http://127.0.0.1:20034/health
```

Expected:

```
{"status":"ok"}
```

## Run the test

From the project root:

```
python test/d4rt/test_d4rt_tool.py
```

If import fails, use:

```
PYTHONPATH=/data/sjq python test/d4rt/test_d4rt_tool.py
```
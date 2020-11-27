# Run with Kubernetes
First, please refer to [the doc here](./RUN_SCRIPTS.md) for how to run each role with bare metal scripts.
Then, let's describe how to run it with k8s.

### docker
See [the descriptions here](../build_docker/README.md) for building docker.

### start all  
```bash
python render_template.py run_ppo_all.yml.jinja2 | kubectl create -f -
```
Note: `kubectl create -f` means creating from file,
the second single dash `-` indicates a special file the `stdin`,
and finally the pipe operator `|` forwards the output string.

### stop all
```bash
python render_template.py run_ppo_all.yml.jinja2 | kubectl delete -f -
```
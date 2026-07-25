# Cowork 3D Portrait — RunPod Serverless

## Files
- `Dockerfile` — image build (base RunPod PyTorch 2.8/CUDA 12.8, miniforge → tripo310 env, all deps pinned, model weights preloaded, sanity import test)
- `handler.py` — RunPod Serverless SDK wrapper. Accepts base64 image + optional mesh, returns base64 GLB + relief cut + transform sidecar
- `pipeline_v14.py`, `pipeline_v11_depth_umeyama.py`, `module_a_retune.py`, `module_e_texture.py` — pipeline code

## Build + push

### Option A: Build locally on Cowork-Host (Docker Desktop required)
```powershell
cd C:\ClaudeProjects\opcreative-3d-portrait\serverless
docker build -t <dockerhub-user>/cowork-3d-v14:latest .
docker push <dockerhub-user>/cowork-3d-v14:latest
```
Build size: ~30GB local, ~15GB pushed (weights compress). Takes 20-40 min first time (weight downloads inside container).

### Option B: RunPod Hub deploy from GitHub
1. Push this folder to a public GitHub repo
2. RunPod dashboard → Serverless → New Endpoint → "Deploy from GitHub"
3. Enter repo URL + branch + Dockerfile path
4. RunPod builds + pushes internally

## Deploy Serverless endpoint

Dashboard → Serverless → New Endpoint:
- **Image**: `<dockerhub-user>/cowork-3d-v14:latest`
- **GPU**: RTX 4090 (fallback: RTX 5090, RTX A6000)
- **Min workers**: 0 (scale to zero — no idle cost)
- **Max workers**: 1 (Kent solo user)
- **Idle timeout**: 30s
- **Execution timeout**: 600s
- **Container disk**: 40GB

## Invoke

```bash
# Encode input
IMG_B64=$(base64 -w0 person_2.png)
MESH_B64=$(base64 -w0 person_2.glb)

# Sync call
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"image_b64\": \"$IMG_B64\", \"mesh_b64\": \"$MESH_B64\", \"run_module_cd\": true, \"run_module_e\": true}}" \
  > result.json

# Decode outputs
jq -r '.output.glb_b64' result.json | base64 -d > person_2_v14.glb
jq -r '.output.relief_b64' result.json | base64 -d > person_2_v14_face_relief_only.glb
jq -r '.output.transform' result.json > person_2_v14_transform.json
```

## Cost per invocation

RTX 4090 Serverless: ~$0.00019/sec (~$0.68/hr).
- Cold start (image pull): 15-45s
- Warm run: 30-60s pipeline
- Typical: $0.01-0.05 per iteration.

Scale to zero = no cost when idle.

## Deploy via RunPod GraphQL (automation)

```graphql
mutation {
  saveEndpoint(input: {
    name: "cowork-3d-v14"
    templateId: "<template-id>"
    gpuIds: "NVIDIA GeForce RTX 4090"
    workersMin: 0
    workersMax: 1
    idleTimeout: 30
    scalerType: "QUEUE_DELAY"
    scalerValue: 4
  }) { id }
}
```

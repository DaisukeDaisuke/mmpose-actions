"""Export the trained 133-point SDPose model on GitHub Actions.

The heatmap head below follows SDPose's MIT-licensed HeatmapHead.py.
Copyright (c) 2025 S. Liang. See LICENSE-SDPose.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

SOURCE = 'teemosliang/SDPose-Wholebody'
REVISION = '0c8153e39458bc8ae87fb68790c141bf0f03c11b'
CODE_REVISION = '90517ab286390131199630394e546d9a14e4c1c1'


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def describe(folder, component, policy):
    import onnx
    model_path = folder / 'model.onnx'
    onnx.checker.check_model(str(model_path))
    graph = onnx.load(str(model_path), load_external_data=False)
    def contract(value):
        return {'name': value.name, 'type': value.type.tensor_type.elem_type,
                'shape': [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]}
    report = {'component': component, 'policy': policy, 'base_model': SOURCE,
              'baseRevision': REVISION, 'sourceCodeRevision': CODE_REVISION,
              'buildCommit': os.environ.get('GITHUB_SHA'), 'keypoints': 133,
              'dimensions': 2, 'inputs': list(map(contract, graph.graph.input)),
              'outputs': list(map(contract, graph.graph.output)),
              'browserGPUValidated': False, 'accuracyValidated': False,
              'speedValidated': False,
              'files': [fingerprint(p) for p in sorted(folder.iterdir()) if p.is_file()]}
    (folder / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def export(component, folder):
    import torch
    from torch import nn
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.set_num_threads(2)
    torch.manual_seed(0)
    patterns = ['vae/*'] if component == 'vae' else ['unet/*', 'decoder/*', 'text_encoder/*', 'tokenizer/*']
    source = snapshot_download(SOURCE, revision=REVISION, allow_patterns=patterns,
                               cache_dir='.cache/sdpose', max_workers=2)

    class Encoder(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.encoder = vae.encoder
            self.quant_conv = vae.quant_conv

        def forward(self, image, posterior_noise):
            moments = self.quant_conv(self.encoder(image))
            mean, logvar = torch.chunk(moments, 2, dim=1)
            return (mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * posterior_noise) * 0.18215

    class HeatmapHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(640, 640, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(640), nn.SiLU(inplace=True))
            self.conv_layers = nn.Sequential(nn.Conv2d(640, 640, 1),
                                             nn.InstanceNorm2d(640), nn.SiLU(inplace=True))
            self.final_layer = nn.Conv2d(640, 133, 1)

        def forward(self, feature):
            return self.final_layer(self.conv_layers(self.deconv_layers(feature)))

    class UnusedFinalBlock(nn.Module):
        # Wholebody consumes the second-last up-block feature. The final block
        # and noise-prediction head have no path to the pose output.
        has_cross_attention = False
        def __init__(self, count):
            super().__init__()
            self.resnets = nn.ModuleList([nn.Identity() for _ in range(count)])

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class Pose(nn.Module):
        def __init__(self, unet, head, empty_text):
            super().__init__()
            self.unet, self.head = unet, head
            self.register_buffer('empty_text', empty_text)
            self.register_buffer('timestep', torch.tensor(999, dtype=torch.int64))
            task = torch.tensor([[1., 0.]])
            self.register_buffer('task', torch.cat([torch.sin(task), torch.cos(task)], dim=-1))

        def forward(self, latent):
            feature = self.unet(latent, self.timestep, self.empty_text,
                                class_labels=self.task, return_dict=False)[0]
            return self.head(feature)

    if component == 'vae':
        vae = AutoencoderKL.from_pretrained(source, subfolder='vae', local_files_only=True)
        vae.set_attn_processor(AttnProcessor())
        model = Encoder(vae).eval()
        del vae
        inputs = (torch.zeros(1, 3, 1024, 768), torch.zeros(1, 4, 128, 96))
        names, outputs = ['image', 'posterior_noise'], ['latent']
    else:
        tokenizer = CLIPTokenizer.from_pretrained(source, subfolder='tokenizer', local_files_only=True)
        text = CLIPTextModel.from_pretrained(source, subfolder='text_encoder', local_files_only=True).eval()
        tokens = tokenizer('', padding='do_not_pad', max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors='pt').input_ids
        with torch.no_grad():
            empty = text(tokens)[0].detach()
        del text, tokenizer
        gc.collect()
        unet = UNet2DConditionModel.from_pretrained(source, subfolder='unet', local_files_only=True,
                                                   low_cpu_mem_usage=True)
        unet.set_attn_processor(AttnProcessor())
        unet.up_blocks[-1] = UnusedFinalBlock(len(unet.up_blocks[-1].resnets))
        unet.conv_norm_out = None
        unet.conv_out = nn.Identity()
        head = HeatmapHead()
        head.load_state_dict(load_file(str(Path(source) / 'decoder/decoder.safetensors')), strict=True)
        model = Pose(unet, head, empty).eval()
        inputs = (torch.zeros(1, 4, 128, 96),)
        names, outputs = ['latent'], ['heatmaps']
    gc.collect()
    with torch.no_grad():
        # The legacy JIT exporter serializes the >2GiB pose graph before
        # externalizing tensors. Dynamo writes that graph with external data.
        use_dynamo = component == 'pose'
        options = {'optimize': False} if use_dynamo else {'do_constant_folding': False}
        torch.onnx.export(model, inputs, str(folder / 'model.onnx'), input_names=names,
                          output_names=outputs, opset_version=18 if use_dynamo else 17,
                          dynamo=use_dynamo, external_data=True, **options)
    del model
    gc.collect()
    describe(folder, component, 'fp32')


def quantize(component, policy, source, folder):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(source / 'model.onnx'))
    if policy == 'fp16':
        from onnxconverter_common.float16 import convert_float_to_float16
        model = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    else:
        selected = policy.removeprefix('int8-weight-')
        initializers = {w.name: w for w in model.graph.initializer}
        additions, replacements, nodes = [], {}, []
        for index, node in enumerate(model.graph.node):
            is_head = ('/head/' in node.name or node.name.startswith('/head')
                       or any(name.startswith('head.') for name in node.input[1:]))
            eligible = selected == 'all' or selected == 'unet' and component == 'pose' and not is_head or selected == 'head' and component == 'pose' and is_head
            if eligible and node.op_type in {'Conv', 'ConvTranspose', 'MatMul', 'Gemm'} and len(node.input) > 1 and node.input[1] in initializers:
                name = node.input[1]
                if name not in replacements:
                    weight = numpy_helper.to_array(initializers[name])
                    if weight.dtype != np.float32:
                        raise ValueError('Expected FP32 source weights')
                    scale = np.float32(max(float(np.abs(weight).max()) / 127, 1e-12))
                    prefix = f'weight_qdq_{index}'
                    additions += [numpy_helper.from_array(np.clip(np.rint(weight / scale), -127, 127).astype(np.int8), prefix + '_q'),
                                  numpy_helper.from_array(np.asarray(scale), prefix + '_scale'),
                                  numpy_helper.from_array(np.asarray(0, dtype=np.int8), prefix + '_zero')]
                    nodes.append(helper.make_node('DequantizeLinear', [prefix + '_q', prefix + '_scale', prefix + '_zero'], [prefix + '_float'], name=prefix))
                    replacements[name] = prefix + '_float'
                node.input[1] = replacements[name]
            nodes.append(node)
        # Remove a float initializer only when no unquantized consumer uses it.
        used = {name for node in nodes for name in node.input}
        retained = [w for w in model.graph.initializer if w.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained + additions)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        if not replacements:
            raise RuntimeError('No weights matched the requested quantization scope')
    onnx.save_model(model, str(folder / 'model.onnx'), save_as_external_data=True,
                    all_tensors_to_one_file=True, location='weights.bin', size_threshold=1024)
    del model
    gc.collect()
    describe(folder, component, policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['export', 'quantize'])
    parser.add_argument('--component', choices=['vae', 'pose'], required=True)
    parser.add_argument('--policy', default='fp32', choices=['fp32', 'fp16', 'int8-weight-all', 'int8-weight-unet', 'int8-weight-head'])
    parser.add_argument('--source', type=Path, default=Path('base'))
    parser.add_argument('--output', type=Path, default=Path('release'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'export':
        export(args.component, args.output)
    else:
        quantize(args.component, args.policy, args.source, args.output)

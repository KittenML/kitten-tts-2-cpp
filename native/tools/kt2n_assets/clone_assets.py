"""Validate and prepare the original native cloning networks.

Tensor digests pin the original FP32 export of pyannote/embedding and the
production front-end constants. Header ordering and legacy speaker-bank data
may differ; no tensor arithmetic or precision conversion is performed here.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

from .safetensors_raw import SafetensorsFile, sha256_file, write_safetensors

PYANNOTE_REPO = "pyannote/embedding"
PYANNOTE_REVISION = "4db4899737a38b2d618bbd74350915aa10293cb2"
PYANNOTE_CHECKPOINT_SHA256 = "4bcec986de13da7af7ac88736572692359950df63669989c4f78b294934c9089"
# Pinned FP32 tensor shape and raw-byte SHA256, exported by the original native
# oracle_s3side.py and oracle_pyannote.py at source d0fa06b1ebdbb502460196f6b8dd86394e8f6830.
CONTRACTS = {'consts': {'kaldi_mel_banks': ([80, 257], '4696fcf5cd78c83eddf9e6c2300eaf6fcee0d2a1bef3d079b8749a17ee8d777a'),
            'kaldi_povey_400': ([400], '36444c5814867e099a289b76c2a6adc4c7fe721bb64edc54910dc74607a7ca6f'),
            'mel24_basis': ([80, 961], '6ae435f7b8b88c19e3ee46bb8a39b2091883d8371a268561923904bb9d0ec312'),
            'mel24_window': ([1920], '3a8d9fac0e1e526de9bc4a0038e080c7671819d83dbcc28ac206cacf2fbfe1d0'),
            'poly_2_3_taps': ([61], '02f8ddf79873439800d916038c4e714c97eee2c9551bbb98cbf8c992ffe0ffc0'),
            's3_mel_filters': ([128, 201], '2a5f9822897750e047c85dea37cc268d3be0ecfd23c28a5f10da129d99d05afe'),
            's3_window': ([400], 'a63f972e9f978eede31f05b3ffbde1c1931207e94bdfe2d0ca0611ac58b873be'),
            'ta_24_16_kernel_cpu': ([2, 1, 23],
                                    '903d80dfef6c1bc5a1139496a7a352f2a1664894d7330042463f322e208dc5fa'),
            'ta_24_16_kernel_dev': ([2, 1, 23],
                                    'e00f5cca3a2dda45934fa4b7807dd02fc81a5328ea8420cbca981c73cd22ab40')},
 'pyannote': {'embedding.bias': ([512], '5f98e6b275444966b7f8392a3cd3fd0673356f7bddbf9acf890cec37390ee18b'),
              'embedding.weight': ([512, 3000],
                                   '3b2b40ebc7ed2de62aa16b8dcddbdf8b13108d9aac383dc802aabf48c537fd05'),
              'sincnet.conv1d.0.filterbank.band_hz_': ([40, 1],
                                                       '8ecc00e2928f725552854385e424c42cbccd7024f688867094441c0d366afa61'),
              'sincnet.conv1d.0.filterbank.low_hz_': ([40, 1],
                                                      '2835db3ac939fc774340cf47994096da0ec94b6c08a3e8bd39553e40b17d5929'),
              'sincnet.conv1d.0.filterbank.n_': ([1, 125],
                                                 '123e29d3cfa5f7843b09f605dd8db081a5f23e512e75113d271042178b7fdaad'),
              'sincnet.conv1d.0.filterbank.window_': ([125],
                                                      '28f41f909f7e9e16fbfe15ecfc790c9160310266528e7837334a4a8b15bfbfdf'),
              'sincnet.conv1d.0.filters': ([80, 1, 251],
                                           '07308701cd15daf5cc639150079e8e3d5b8eb843776e21cf0c6ef2e51f93aede'),
              'sincnet.conv1d.1.bias': ([60], '113d5664fdb388291cca094cca105c3ecd354b19cc6ea962a7cc81ad1456a8b6'),
              'sincnet.conv1d.1.weight': ([60, 80, 5],
                                          'fe207e8b8bb4affa1960dbeafd6761fbe520719d70272c58ad115ad62873da10'),
              'sincnet.conv1d.2.bias': ([60], '346cb52503b19579831049b77ba2c6d7f61553589832c7b84e5d0653e93c69eb'),
              'sincnet.conv1d.2.weight': ([60, 60, 5],
                                          'ab32b0dd48b6e4a80edacc498b3e4b040e1023bc6c106d56bbd8aa88a28578db'),
              'sincnet.norm1d.0.bias': ([80], '6b8bc2259b08f2b742b7e3a41ce27c83785189916acf3053c34bc951f840d882'),
              'sincnet.norm1d.0.weight': ([80],
                                          '143e7e17a9464ec374731553d2118cf4a7095bb95dfb0f2517159e1d5f1f2ae1'),
              'sincnet.norm1d.1.bias': ([60], 'e5ccc37adbf051a9e04b2e54b3ca4ee75414aae7181960aef3b060a6577ab654'),
              'sincnet.norm1d.1.weight': ([60],
                                          'b1c6c1165761fb7112c4be43ac10b1660577b637bd6fd9f5aa63c580e2c6d179'),
              'sincnet.norm1d.2.bias': ([60], 'a901e901fa7da05a7d2b23c4116d90194a5b8d373a8cd76693152fc48ab688ee'),
              'sincnet.norm1d.2.weight': ([60],
                                          '5b95c1e5d0753cfa58cbffd3ca5ed36d8a400ab0454c8690e53f7eefdd4c4b50'),
              'sincnet.wav_norm1d.bias': ([1], '718fc84f9e7cf9dd4d88a321f21a1216150fecdefce2bccb8b16c15a672bc20d'),
              'sincnet.wav_norm1d.weight': ([1],
                                            '984bf21328be4acd8fa8fd28db0b79932301ca0f518e89f090c1daa023f80166'),
              'tdnns.0.bias': ([512], '8c1d6e5cc7f154a57201f96ccedd5fc488ceb16e5197066371c2120e5275f3e5'),
              'tdnns.0.weight': ([512, 60, 5], '1d2c894231a675d64bf9dbe37421cd8cb24478f64b742f980f5266e21a891e3d'),
              'tdnns.11.bias': ([512], '27f8712468dd8e3715f2a67bcd84862e24eee52779c3b56907678fa7cc10c30a'),
              'tdnns.11.running_mean': ([512], '9ea2a94fec9269e6944f04771f7683684995515d016cb02f6d5d171b0b33a2ca'),
              'tdnns.11.running_var': ([512], '2d868f6966da718e2e55948a22e4f0f86423ad09908c4fdbc536181f9e20b709'),
              'tdnns.11.weight': ([512], '380df3b96278f1e7880e817eb36bb69dcc63dceb6a1c662db9ae9d193ad881dd'),
              'tdnns.12.bias': ([1500], 'e13cc7835969ae70923cabdc6f10c7f62b1fefb8c1295365643d1d24f8e4d887'),
              'tdnns.12.weight': ([1500, 512, 1],
                                  'e13ae4d12c2b4aab2d71ade3b97f577a14c64beccbab5f2fc85dc0f04ecc03f9'),
              'tdnns.14.bias': ([1500], 'bde63db36d90f5ea16a9e86e913276696604a62e83edcc213decdd2f29c9d9a3'),
              'tdnns.14.running_mean': ([1500],
                                        'c7c2f1a4f9070309d50606eb17c6598dcb37ff38e52882e040b59ead88f54ebd'),
              'tdnns.14.running_var': ([1500], '9985ffd6b7a14bb98510d442ac41592a448289a962e7bb91f8719cafca048f6f'),
              'tdnns.14.weight': ([1500], '5eb7c31002fae7860ebed77bb66e6e7227ea2873ded637a5f9b3f17a3618d0b5'),
              'tdnns.2.bias': ([512], 'ff1ef69b92c210b87cbd81d4225de08b0bccfb60c5bbc0d3bd07835d5c53ac55'),
              'tdnns.2.running_mean': ([512], 'd5ee3bf58e53c24794a771544fe95c3b06d2b08165bf3423ff389c78d8e5e4d7'),
              'tdnns.2.running_var': ([512], '87bdb4908ea1ec2ba69195762aa28f95efd2adfc83781e962b710f96632226a0'),
              'tdnns.2.weight': ([512], '0f2af1a1199f3634675082979769062faa318a8e3725f52a12efd04a9bdb929b'),
              'tdnns.3.bias': ([512], '51294bcda5c435a4e25b2706e8c6b38d597f636e77062e5534a52abaae9ab60e'),
              'tdnns.3.weight': ([512, 512, 3],
                                 '04691952c568eca18825883e17c065a570329963b39208e761e8d96e58361452'),
              'tdnns.5.bias': ([512], '6955cb2f3b5176abe8395d09bb9979b3bae2d49b756ef0fd5e8cdcaa44280cf0'),
              'tdnns.5.running_mean': ([512], 'e0ec7ededb87bf507bea7faffb1ca2947b7c60139160332ae7a403fd5c52b35e'),
              'tdnns.5.running_var': ([512], '6b5bacaa47950f7c477b23485d1cd04779e56f64f25d4219db7e6c8351aa1168'),
              'tdnns.5.weight': ([512], '764c73b513086ffeab3a70c3f08772926f73c8c89aacc94a43c8cba97d3e79c4'),
              'tdnns.6.bias': ([512], 'e705bc980d1df2ba41ce1a92b2b5c8aaffe379e08a5b15c59d13db9ab29a50a1'),
              'tdnns.6.weight': ([512, 512, 3],
                                 '83f148f9dc06df8155857d00c8b0913ad97abec914531e0b781390f0ed13cbce'),
              'tdnns.8.bias': ([512], '9812cfe80358c5d5406602ae58832b2ec852cc8a9c510783c27c70fd1deb9945'),
              'tdnns.8.running_mean': ([512], '2a1745d8b0a53c24c3f69ab390adafdfb54256e436c7bf6bb7be9714d6947c24'),
              'tdnns.8.running_var': ([512], '5f4102030880c107a0235839d31ea3e2841728f3b2b498c11ccd79632155fe10'),
              'tdnns.8.weight': ([512], '0b8c64abad8be78c673f647be64d689af4680db391fb2dda396ce49399c697f0'),
              'tdnns.9.bias': ([512], '9bb23729436a8308a638e379f1435b6067ba697cb1f38245dcb31786b1b32016'),
              'tdnns.9.weight': ([512, 512, 1],
                                 '9167ff2b16471f6ee5bd86589dffb8be13ea413b738b2775d81df5ba6d346cef')}}


def checked_tensors(path: Path, kind: str) -> dict:
    """Accept only the pinned network/constants; discard legacy bank metadata."""
    contract = CONTRACTS[kind]
    with SafetensorsFile(path) as source:
        actual = {name for name in source.by_name if not name.startswith("bank.")}
        if actual != set(contract):
            raise ValueError(f"clone {kind}: tensor names do not match the pinned export")
        tensors = {}
        for name, (shape, digest) in contract.items():
            info = source.by_name[name]
            if info.dtype != "F32" or list(info.shape) != shape or source.tensor_sha256(name) != digest:
                raise ValueError(f"clone {kind}: tensor checksum/shape mismatch: {name}")
            tensors[name] = ("F32", info.shape, bytes(source.raw(name)))
    return tensors


def prepare_clone_files(acoustic: Path, consts: Path, pyannote: Path, destination: Path) -> dict:
    """Copy the pinned native weights, stripping the legacy catalogue bank."""
    from .convert_s3gen import PINNED_SHA256, PINNED_REVISION
    import shutil
    if sha256_file(acoustic) != PINNED_SHA256:
        raise ValueError("clone acoustic source checksum mismatch")
    tensors = {"consts": checked_tensors(consts, "consts"),
               "pyannote": checked_tensors(pyannote, "pyannote")}
    if destination.exists():
        raise ValueError("clone destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clone-prepare-", dir=destination.parent) as scratch:
        stage = Path(scratch) / "cloning"
        stage.mkdir()
        shutil.copyfile(acoustic, stage / "s3gen.safetensors")
        write_safetensors(stage / "voicebuild_consts.safetensors", tensors["consts"],
                         {"ta_width": "10", "source": "native-cloning-constants-v1"})
        write_safetensors(stage / "voicebuild_pyannote.safetensors", tensors["pyannote"],
                         {"model": f"{PYANNOTE_REPO}@{PYANNOTE_REVISION}"})
        stage.rename(destination)
    return {"acoustic_repo": "ResembleAI/chatterbox-turbo", "acoustic_revision": PINNED_REVISION,
            "acoustic_sha256": PINNED_SHA256, "pyannote_repo": PYANNOTE_REPO,
            "pyannote_revision": PYANNOTE_REVISION, "pyannote_checkpoint_sha256": PYANNOTE_CHECKPOINT_SHA256,
            "pyannote_export_sha256": sha256_file(pyannote), "constants_export_sha256": sha256_file(consts),
            "tensor_validation": "native-cloning-fp32-v1", "legacy_speaker_bank": "removed",
            "speaker_bank": "voices.kt2n builtin raw_speaker rows", "model_license": "MIT",
            "pyannote_access": "requires acceptance of the upstream model access conditions"}

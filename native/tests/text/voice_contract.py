"""Standalone contract for the vendored voice backend, without CUDA/OpenFst."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'src/text/voice'
    manifest = json.loads((source / 'manifest.json').read_text())
    for filename, digest in manifest['files'].items():
        assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == digest, filename
    lib = ctypes.CDLL(sys.argv[1])
    lib.kitten_voice_warm.restype = ctypes.c_int
    assert lib.kitten_voice_warm() == 0
    lib.kitten_voice_normalize.restype = ctypes.c_void_p
    lib.kitten_voice_normalize.argtypes = [ctypes.c_char_p, ctypes.c_longlong,
                                         ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong)]
    lib.kitten_voice_free.argtypes = [ctypes.c_void_p]

    def render(text, language):
        data = text.encode()
        length = ctypes.c_longlong()
        ptr = lib.kitten_voice_normalize(data, len(data), language.encode(), ctypes.byref(length))
        assert ptr
        try:
            return ctypes.string_at(ptr, length.value).decode()
        finally:
            lib.kitten_voice_free(ptr)

    expected = {'en': 'twelve', 'ar': 'اثنا عشر', 'zh': '十二', 'nl': 'twaalf',
                'fr': 'douze', 'de': 'zwölf', 'hi': 'बारह', 'id': 'dua belas',
                'it': 'dodici', 'ja': '十二', 'ko': '십이', 'pl': 'dwanaście',
                'pt': 'doze', 'ru': 'двенадцать', 'es': 'doce', 'tr': 'on iki'}
    def check(language):
        assert render('12', language) == expected[language], language
        protected = 'v1.2.3 AB12.34 192.168.1.2 1990s3.5'
        assert render(protected, language) == protected, language
        assert render('2027-02-30', language) == '2027-02-30', language
        contact = render('user12@gmail.com', language)
        assert contact != 'user12@gmail.com' and '@' not in contact and '12' not in contact, language
        assert render(contact, language) == contact, language
        for index in (0, 1, 2, 9, 10, 42, 123):
            marker = f'\ue000{index}\ue001'
            assert render(marker, language) == marker, language
            assert render(f'{marker} 12 {marker}', language) == f'{marker} {expected[language]} {marker}', language
        markers = ''.join(f'\ue000{i}\ue001' for i in range(21))
        assert render(markers, language) == markers, language
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, list(expected) * 4))
    assert render('总金额是USD 1,234.56。', 'zh') == '总金额是一千二百三十四美元五十六美分。'
    assert render('Harga tiketnya Rp12.500,50.', 'id') == 'Harga tiketnya dua belas ribu lima ratus rupiah lima puluh sen.'
    assert render('14:05', 'ko') == '십사 시 오 분'
    print('16 language contracts, contacts, reviewed amounts, protected values, invalid dates and source hashes passed')


if __name__ == '__main__':
    main()

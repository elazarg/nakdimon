import logging
import pathlib
from functools import lru_cache
from typing import Any

from nakdimon import utils, dataset, hebrew
from nakdimon.config import MAIN_MODEL

def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for the built-in Nakdimon .h5 predictor. "
            "Install it with `pip install 'nakdimon[tf]'`."
        ) from exc

    if tf.config.set_visible_devices([], 'GPU'):
        logging.warning('No GPU available.')

    return tf


@lru_cache()
def load_cached_model(m: pathlib.Path | str) -> Any:
    if isinstance(m, str):
        return load_cached_model(pathlib.Path(m))
    assert isinstance(m, pathlib.Path)
    tf = _require_tensorflow()
    model = tf.keras.models.load_model(m, custom_objects={'loss': None})
    return model


def merge_unconditional(texts, tnss, nss, dss, sss):
    res = []
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            sentence.append(dataset.dagesh_table.indices_char[d] if hebrew.can_dagesh(t) else '\uFEFF')
            sentence.append(dataset.sin_table.indices_char[s] if hebrew.can_sin(t) else '\uFEFF')
            sentence.append(dataset.niqqud_table.indices_char[n] if hebrew.can_niqqud(t) else '\uFEFF')
        res.append(''.join(sentence))
    return res


def predict(text: str, model_or_model_path: Any | str = MAIN_MODEL, maxlen=10000) -> str:
    tf = _require_tensorflow()
    if isinstance(model_or_model_path, (pathlib.Path, str)):
        model_or_model_path = load_cached_model(model_or_model_path)
    if not isinstance(model_or_model_path, tf.Module):
        raise TypeError(f'Expected str or tf.Module, got {type(model_or_model_path)}')
    model = model_or_model_path
    data = dataset.Data.from_text(hebrew.iterate_dotted_text(text), maxlen)
    prediction = model.predict(data.normalized)
    [actual_niqqud, actual_dagesh, actual_sin] = [dataset.from_categorical(prediction[0]), dataset.from_categorical(prediction[1]), dataset.from_categorical(prediction[2])]
    actual = merge_unconditional(data.text, data.normalized, actual_niqqud, actual_dagesh, actual_sin)
    return ' '.join(actual).replace('\ufeff', '').replace('  ', ' ').replace(hebrew.RAFE, '')


def main(input_path='-', output_path='-'):
    with utils.smart_open(input_path, 'r', encoding='utf-8') as f:
        text = hebrew.remove_niqqud(f.read())
    text = predict(text)
    with utils.smart_open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

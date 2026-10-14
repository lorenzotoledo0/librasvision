"""Modelo: LSTM empilhada com softmax, parametrizada pelo YAML (secao 'modelo')."""

from __future__ import annotations


def construir_modelo(cfg, comprimento: int, dim: int, n_classes: int):
    import tensorflow as tf

    m = cfg["modelo"]
    camadas = list(m["camadas_lstm"])
    if not camadas:
        from .erros import ErroConfig

        raise ErroConfig("modelo.camadas_lstm precisa de pelo menos uma camada.")
    entrada = tf.keras.Input(shape=(comprimento, dim), name="janela")
    x = entrada
    for i, u in enumerate(camadas):
        ultima = i == len(camadas) - 1
        # ativacao padrao (tanh) mantem o kernel otimizado e e mais estavel que ReLU em LSTM
        x = tf.keras.layers.LSTM(
            int(u),
            return_sequences=not ultima,
            recurrent_dropout=float(m.get("dropout_recorrente", 0.0)),
            name=f"lstm_{i + 1}",
        )(x)
        if m.get("dropout", 0) > 0:
            x = tf.keras.layers.Dropout(float(m["dropout"]), name=f"dropout_{i + 1}")(x)
    if int(m.get("densa", 0)) > 0:
        x = tf.keras.layers.Dense(int(m["densa"]), activation="relu", name="densa")(x)
    saida = tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax")(x)
    return tf.keras.Model(entrada, saida, name="librasvision_lstm")


def compilar(modelo, taxa_aprendizado: float) -> None:
    import tensorflow as tf

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(taxa_aprendizado),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

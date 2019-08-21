import tensorflow as tf


def get_pyramids(inputs, model_name, pyramid_names):
    backbone_fn = getattr(tf.keras.applications, model_name)
    base_model = backbone_fn(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    pyramids = [base_model.get_layer(name).output 
                for name in pyramid_names]
    model = tf.keras.Model(input=base_model.input, outputs=pyramids)
    return model(inputs)


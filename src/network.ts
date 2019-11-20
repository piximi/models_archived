import * as tensorflow from '@tensorflow/tfjs';
import * as _ from 'lodash';

const createMobileNet = async (
  classes: number,
  freeze: boolean
) => {
  const resource =
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

  const mobilenet = await tensorflow.loadLayersModel(resource);

  const layerName = 'conv_pw_13_relu';
  const layer = mobilenet.getLayer(layerName);

  const backbone = tensorflow.model({
    inputs: mobilenet.inputs,
    outputs: layer.output
  });

  if (freeze) {
    backbone.layers.forEach(function(l){
      l.trainable = false;
    })
  }

  const a = tensorflow.layers.globalAveragePooling2d({
    inputShape: backbone.outputs[0].shape.slice(1)
  });

  const b = tensorflow.layers.reshape({
    targetShape: [1,1,backbone.outputs[0].shape[3]]
  });

  const c = tensorflow.layers.dropout({
    rate: 0.001
  });

  const d = tensorflow.layers.conv2d({
    filters: classes,
    kernelSize: [1,1]
  });

  const e = tensorflow.layers.reshape({
    targetShape: [classes]
  });

  const f = tensorflow.layers.activation({
    activation: 'softmax'
  });

  const config = {
    layers: [...backbone.layers, a, b, c, d, e, f]
  };

  const model = tensorflow.sequential(config);

  return model;
};

const createModel = async (numberOfClasses: number) => {
  const model = tensorflow.sequential();
  model.add(
    tensorflow.layers.conv2d({
      inputShape: [224, 224, 3],
      kernelSize: 3,
      filters: 16,
      activation: 'relu',
      kernelInitializer: 'ones'
    })
  );
  model.add(tensorflow.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(
    tensorflow.layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: 'relu',
      kernelInitializer: 'ones'
    })
  );
  model.add(tensorflow.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(
    tensorflow.layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: 'relu',
      kernelInitializer: 'ones'
    })
  );
  model.add(tensorflow.layers.flatten());
  model.add(
    tensorflow.layers.dense({
      units: 32,
      activation: 'relu',
      kernelInitializer: 'ones'
    })
  );
  model.add(
    tensorflow.layers.dense({
      units: numberOfClasses,
      activation: 'relu',
      kernelInitializer: 'ones'
    })
  );
  model.add(
    tensorflow.layers.dense({ units: numberOfClasses, activation: 'softmax' })
  );
  return model;
};

const getArgs = (batchSize: number, epochs: number) => {
  const arg = {
    batchSize: batchSize,
    callbacks: {
      onTrainBegin: async (logs?: tensorflow.Logs | undefined) => {
        console.log(`onTrainBegin`);
      },
      onTrainEnd: async (logs?: tensorflow.Logs | undefined) => {},
      onEpochBegin: async (
        epoch: number,
        logs?: tensorflow.Logs | undefined
      ) => {
        console.log(`onEpochBegin ${epoch}`);
      },
      onEpochEnd: async (epoch: number, logs?: tensorflow.Logs | undefined) => {
        if (logs) {
          console.log(`onEpochEnd ${epoch}, loss: ${logs.loss}`);
        }
        // if (stopTraining) {
        //   model.stopTraining = true;
        // }
      },
      onBatchBegin: async (
        batch: number,
        logs?: tensorflow.Logs | undefined
      ) => {
        console.log(`onBatchBegin ${batch}`);
      },
      onBatchEnd: async (batch: number, logs?: tensorflow.Logs | undefined) => {
        console.log(`onBatchEnd ${batch}`);
      }
    },
    epochs: epochs
  };
  return getArgs;
};

export { createModel, createMobileNet, getArgs };

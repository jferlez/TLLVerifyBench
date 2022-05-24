
import os
import sys
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import tensorflow as tf
import onnx
import re
import pickle
import zipfile

import onnxruntime as rt
providers = ['CPUExecutionProvider']
# m = rt.InferenceSession(tll.onnxModel.SerializeToString(), providers=providers)




if __name__ == '__main__':

    if len(sys.argv) >= 2:
        rs = RandomState(MT19937(SeedSequence(int(sys.argv[1]))))

    with open('tllBench_database.p','rb') as fp:
        tllBenchDatabase = pickle.load(fp)

    timeout = int(600) # seconds
    numSizes = len(tllBenchDatabase['N'].keys())
    numInstances = (6 * 3600)//(numSizes * timeout) # Total number of instances to create a maximum run time of <= 6 hours
    vnnlibPath = './vnnlib/'

    inputExtents = [
        [-2,2], \
        [-2,2]
    ]
    numEdgeSamples = 100
    edgeSamples = [np.linspace(ext[0],ext[1],numEdgeSamples) for ext in inputExtents]

    for size in tllBenchDatabase['N']:
        for instIdx in range(numInstances):

            inputSamples = np.vstack(list(map(lambda x: x.flatten(),list(np.meshgrid(*edgeSamples))))).T.astype(np.float32)

            with zipfile.ZipFile('./onnx_compressed/' + tllBenchDatabase["N"][size][instIdx]["baseFileName"] + '.onnx.zip', 'r') as zip_ref:
                zip_ref.extractall('./onnx/')
            onnxModel = onnx.load('./onnx/' + tllBenchDatabase["N"][size][instIdx]["baseFileName"] + '.onnx')
            m = rt.InferenceSession('./onnx/' + tllBenchDatabase["N"][size][instIdx]["baseFileName"] + '.onnx', providers=providers)
            
            outputSamples = m.run(onnxModel.graph.node[-1].output, {"input": inputSamples})[0]
            outputExtents = [np.min(outputSamples), np.max(outputSamples)]
            outputWidth = (outputExtents[1]-outputExtents[0])
            outputCenter = (outputExtents[1]+outputExtents[0])/2

            propThresh = 2*outputWidth*(np.random.random_sample() - 0.5) + outputCenter
            propDirection = '>=' if (np.random.random_sample() >= 0.5) else '<='

            with open(f'./vnnlib/property_N={size}_{instIdx}.vnnlib','w') as fp:
                # Declare the inputs
                for i in range(len(inputExtents)):
                    fp.write(f'(declare-const X{i} Real)\n')
                # Declare the output
                fp.write(f'(declare-const Y0 Real)\n\n')
                # Write the input constraints:
                fp.write('; Input Constraints:\n')
                for i in range(len(inputExtents)):
                    fp.write(f'(assert (<= X{i} {inputExtents[i][1]}))\n')
                    fp.write(f'(assert (>= X{i} {inputExtents[i][0]}))\n')
                
                # Write the output property:
                fp.write('\n; Output property:')
                fp.write(f'\n; Min output sample: {outputExtents[0]}. Max output sample: {outputExtents[1]}\n')
                fp.write(f'(assert ({propDirection} Y0 {propThresh}))\n')

            with open('tllBench_instances.csv','a') as fp:
                fp.write(f'./onnx/{tllBenchDatabase["N"][size][instIdx]["baseFileName"]}.onnx,./vnnlib/property_N={size}_{instIdx}.vnnlib,{timeout}\n')



            
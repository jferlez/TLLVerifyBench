import sys
pths = [
    '/Users/james/Dropbox/Research/Postdoc Yasser/code/FastBATLLNN', \
    '/Users/james/Dropbox/Research/Postdoc Yasser/code/FastBATLLNN/HyperplaneRegionEnum', \
    '/Users/james/Dropbox/Research/Postdoc Yasser/code/FastBATLLNN/TLLnet', \
    '/Users/james/Dropbox/Research/Postdoc Yasser/code/TLLVerifyBench/nnenum' \
]
for pth in pths:
    if not pth in sys.path:
        sys.path.append(pth)

import numpy as np
import tensorflow as tf
# We don't get agreement with existing sample database without setting Keras internal floats to 64 bit
from keras import backend as K
K.set_floatx('float64')

import TLLnet
from importlib import reload

import onnx
import onnxruntime as rt
providers = ['CPUExecutionProvider']
import re
import pickle


if __name__ == '__main__':

    with open('sizeVsTime_n2_input.p','rb') as fp:
        oldDatabase = pickle.load(fp)
    
    for sizeIdx in range(len(oldDatabase)):
        for instIdx in range(len(oldDatabase[sizeIdx])):
            instDict = oldDatabase[sizeIdx][instIdx]
            print(f'Exporting size index {sizeIdx} (N={instDict["N"]}), instance {instIdx}')
            tll = TLLnet.TLLnet(input_dim=instDict['n'], output_dim=instDict['m'], linear_fns=instDict['N'], uo_regions=instDict['M'], dtypeKeras=tf.float32)
            tll.setLocalLinearFns( \
                    [ \
                        [llf[0].T, llf[1]]
                        for llf in instDict['TLLparameters']['localLinearFunctions'] \
                    ]
                )
            tll.setSelectorSets( \
                    [ \
                        [ set(np.nonzero(sMat)[0]) for sMat in sMats ]
                        for sMats in instDict['TLLparameters']['selectorMatrices'] \
                    ]
                )
            tll.createKeras(incBias=True,flat=True)
            allSelectors = tll.getKerasAllSelectors()
            for k1 in range(len(instDict['TLLparameters']['selectorMatrices'])):
                for k2 in range(len(instDict['TLLparameters']['selectorMatrices'][k1])):
                    assert set(np.nonzero(instDict['TLLparameters']['selectorMatrices'][k1][k2])[0]) == set(np.nonzero(allSelectors[k1][k2])[0]), f'selector mismatch size {sizeIdx}, instance {instIdx}: {instDict["TLLparameters"]["selectorMatrices"][k1][k2]} , {allSelectors[k1][k2]}'

            # This assertion will fail unless Keras' internal floats are set to 64-bit with `K.set_floatx('float64')` (see above)
            # assert np.allclose(tll.model.predict(instDict['samples']['input'].astype(np.float64)), instDict['samples']['output'].astype(np.float64)), \
            #     f'Sample outputs failed for size index {sizeIdx}, instance index {instIdx}: {np.max(np.abs(tll.model.predict(instDict["samples"]["input"].astype(np.float64)) - instDict["samples"]["output"].astype(np.float64)))}'
            # ASSERTION VERIFIED (run using the setting above: hence, these networks really are doing the right thing, and they have been imported correctly)

            tll.save(fname=f'../tll/tllBench_n={instDict["n"]}_N=M={instDict["N"]}_m={instDict["m"]}_instance_{sizeIdx}_{instIdx}.tll')
            
            tll.exportONNX(fname=f'../onnx/tllBench_n={instDict["n"]}_N=M={instDict["N"]}_m={instDict["m"]}_instance_{sizeIdx}_{instIdx}.onnx')
            
            # m = rt.InferenceSession(tll.onnxModel.SerializeToString(), providers=providers)
            # polytopeOutputSamples = m.run(tll.onnxOutputs, {"input": instDict['samples']['input'].astype(np.float32)})

            instDict['inputPoly']['samples'] = instDict['samples']['input']
            with open(f'../input_polytopes/polytope_n={instDict["n"]}_instance_{sizeIdx}_{instIdx}.p','wb') as fp:
                pickle.dump(instDict['inputPoly'], fp)
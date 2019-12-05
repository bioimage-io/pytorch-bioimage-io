import numpy
import pytest

from torchbio.transformations.normalizations import NormalizeZeroMeanUnitVariance

testdata = [
    ([(3, 4)], [(3, 4)]),
]


@pytest.mark.parametrize("ipt_shapes,out_shapes", testdata)
def test_shape(ipt_shapes, out_shapes):
    trf = NormalizeZeroMeanUnitVariance()
    ipt_tensors = [numpy.empty(s) for s in ipt_shapes]
    out_tensors = trf.apply(*ipt_tensors)
    assert all(out.shape == o_shape for out, o_shape in zip(out_tensors, out_shapes))

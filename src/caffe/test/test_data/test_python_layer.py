#!/usr/bin/env python

class MyLayer(object):
    """Simple layer that multiplies input by ten."""

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels,
                bottom[0].height, bottom[0].width)

    def forward(self, bottom, top):
        top[0].data[...] = 10 * bottom[0].data

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = 10 * top[0].diff

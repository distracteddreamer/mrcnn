import tensorflow as tf 

from collections import namedtuple
from easydict import EasyDict

BoxListTuple = namedtuple('BoxList', ['boxes', 'scores', 'target_boxes', 'target_labels', 'anchors'])

class BoxList(EasyDict):

    def __setitem__(self, *args, **kwargs):
        #Immutable
        raise AttributeError('BoxList has no attribute __setitem__')

    def map(self, fn, *args, **kwargs):
        fields, arrs = zip(*self.items())
        def _fn(x, *args, **kwargs):
            x = BoxList(dict(zip(fields, x)))
            x_new = fn(x, *args, **kwargs)
            return [x_new[field] for field in fields]
        arrs_new = tf.map_fn(elems=arrs, fn=lambda x: _fn)
        return BoxList(dict(zip(fields, arrs_new)))

    def concat(self, other):
        if not (set(self) == set(other)):
            raise ValueError('Can only concatenate BoxLists with the same fields')
        _fields = {}
        for field in self:
            _fields[field] = tf.concat(self[field], other[field])
        return BoxList(_fields)

    def masked_select(self, keep):
        _fields = {}
        for field, arr in self.items():
            _fields[field] = tf.boolean_mask(arr, keep)
        return BoxList(_fields)

    def pad(self, pad_size):
        _fields = {}
        for field, arr in self.items():
            padding = tf.concat([
                [[0, pad_size]],
                tf.zeros([tf.rank(arr)-1, 2])], axis=0)
            _fields[field] = tf.pad(arr, padding)
        return BoxList(_fields)

    def select(self, inds):
        _fields = {}
        for field, arr in self.items():
            _fields[field] = tf.gather(arr, inds)
        return BoxList(_fields)


# def apply_to_batch(fn):
#     # fn should return a boxlist
#     def _fn(boxlist, *args, **kwargs):
#         result = tf.map_fn(lambda x: fn(x, *args, **kwargs), boxlist.to_namedtuple())
#         return BoxList(result)
#     return _fn

def topk_select(boxlist):
    keep = tf.nn.top_k(boxlist.scores).indices
    boxlist = boxlist.select(keep)
    return boxlist

def nms_select(boxlist, max_output_size, iou_threshold):
    keep = tf.image.non_max_suppression(boxlist.boxes, boxlist.scores,
            max_output_size=max_output_size, 
            iou_threshold=iou_threshold)
    boxlist = boxlist.select(keep)
    boxlist = boxlist.pad(max_output_size)
    return boxlist




# class BoxList(BoxListTuple):

#     def concat(self, other):
#         return concat_arr(zip(self, other, self._fields, other._fields))

#     def select(self, inds):
#         return select_arr(self, inds)

#     def trim(self, keep):
#         return trim_arr(self, keep)

#     def pad(self, max_size):
#         pad_size = max_size - tf.shape(self.boxes)[0]
#         return pad_arr(self, pad_size)



# def apply_to_boxlist(fn, arrs, *args, **kwargs):
#     def _fn(arrs, *args, **kwargs):
#         return BoxList(*[fn(arr, *args, **kwargs) for arr in arrs])
#     return _fn

# @apply_to_boxlist
# def concat_arr(arr):
#     elem1, elem2, name1, name2 = arr
#     assert name1 == name2
#     return tf.concat([elem1, elem2], axis=0)

# @apply_to_boxlist
# def select_arr(arr, inds):
#     return tf.gather(arr, inds)

# @apply_to_boxlist
# def trim_arr(arr, keep):
#     return tf.boolean_mask(arr, keep)

# @apply_to_boxlist
# def pad_arr(arr, pad_size):
#     padding = tf.concat([
#         [[0, pad_size]],
#         tf.zeros([tf.rank(arr)-1, 2])], axis=0)
#     return tf.pad(arr, padding)


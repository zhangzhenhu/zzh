from .lcs import pyLCS, fast_length
from zzh.encoder import LabelEncoder
from zzh.text import str_clength


class LCS(pyLCS):

    def run(self, a, b):
        self.content_a = a
        self.content_b = b

        self.encoder = LabelEncoder()
        self.encoder.fit(a, b)
        ae = self.encoder.transform(a)
        be = self.encoder.transform(b)
        super(LCS, self).run(ae, be)
        return self.length

    def align(self, padding="_"):
        # self.encoder.fit([padding])
        # padding_id = self.encoder.transform([padding])[0]

        a_pos, b_pos = self.sequence_position()
        a_ = []
        b_ = []

        pre_a = 0
        pre_b = 0
        # padding_count = 0
        for k in range(self.length):
            cur_a = a_pos[k]
            cur_b = b_pos[k]

            a_sub = self.content_a[pre_a:cur_a]
            b_sub = self.content_b[pre_b:cur_b]
            a_.extend(a_sub)
            b_.extend(b_sub)

            pre_a = cur_a
            pre_b = cur_b
            a_length = str_clength(a_)
            b_length = str_clength(b_)
            padding_count = abs(a_length - b_length)
            if padding_count == 0:
                continue

            if a_length < b_length:
                a_.extend([padding] * padding_count)
            else:
                b_.extend([padding] * padding_count)
            # print(a_length, ''.join(a_))
            # print(b_length, ''.join(b_))
        # for i in range(len(a_)):
        #     print(a_[i], b_[i])
        # print(' '.join(array2str(a_)))
        # print(' '.join(array2str(b_)))
        # return self.encoder.inverse_transform(a_), self.encoder.inverse_transform(b_)
        return a_, b_


def array2str(ar):
    return [str(x) for x in ar]

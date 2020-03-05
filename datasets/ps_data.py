

class ps_data:

    @property
    def roidb(self):
        return self._roidb

    @property
    def roidb_indexs(self):
        return self._roidb_indexs

    @property
    def train_pedes(self):
        return self._train_pedes

    @property
    def train_pedes_indexs(self):
        return self._train_pedes_indexs

    @roidb.setter
    def roidb(self, val):
        self._roidb=val

    @roidb_indexs.setter
    def roidb_indexs(self, val):
        self._roidb_indexs=val

    @train_pedes.setter
    def train_pedes(self, val):
        self._train_pedes=val

    @train_pedes_indexs.setter
    def train_pedes_indexs(self, val):
        self._train_pedes_indexs=val


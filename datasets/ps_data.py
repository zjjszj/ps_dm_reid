

class ps_data:

    @property
    def train_data(self):
        return self._train_data

    @property
    def indexs(self):
        return self._indexs

    @train_data.setter
    def train_data(self, val):
        self._train_data=val

    @indexs.setter
    def indexs(self, val):
        self._indexs=val





import numpy as np
from scipy.sparse import csr_matrix
from os.path import join, exists


class Dataset:
    def __init__(self, path):
        if exists(join(path, 'data1.txt')):
            self._load_lastfm(path)
        else:
            self._load_standard(path)

        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_users, self.m_items))
        self._allPos = [self.UserItemNet[u].nonzero()[1] for u in range(self.n_users)]
        self._testDict = self._build_dict(self.testUser, self.testItem)

    def _load_lastfm(self, path):
        train = np.loadtxt(join(path, 'data1.txt'), dtype=int) - 1
        test = np.loadtxt(join(path, 'test1.txt'), dtype=int) - 1
        self.trainUser = train[:, 0]
        self.trainItem = train[:, 1]
        self.testUser = test[:, 0]
        self.testItem = test[:, 1]
        self.n_users = 1892
        self.m_items = 4489

    def _load_standard(self, path):
        train_data = self._parse_file(join(path, 'train.txt'))
        test_data = self._parse_file(join(path, 'test.txt'))
        self.trainUser = np.array([u for u, _ in train_data])
        self.trainItem = np.array([i for _, i in train_data])
        self.testUser = np.array([u for u, _ in test_data])
        self.testItem = np.array([i for _, i in test_data])
        all_users = np.concatenate([self.trainUser, self.testUser])
        all_items = np.concatenate([self.trainItem, self.testItem])
        self.n_users = int(all_users.max()) + 1
        self.m_items = int(all_items.max()) + 1

    def _parse_file(self, filepath):
        data = []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    user = int(parts[0])
                    items = [int(x) for x in parts[1:]]
                    data.extend((user, item) for item in items)
                except ValueError:
                    continue
        return data

    def _build_dict(self, users, items):
        d = {}
        for user, item in zip(users, items):
            if user not in d:
                d[user] = []
            d[user].append(item)
        return d

    @property
    def testDict(self):
        return self._testDict

    @property
    def allPos(self):
        return self._allPos

    def getUserPosItems(self, users):
        return [self.UserItemNet[int(u)].nonzero()[1] for u in users]

    @property
    def trainDataSize(self):
        return len(self.trainUser)

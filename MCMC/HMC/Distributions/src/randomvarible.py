class RandomVariable:
    def _size(self):
        raise NotImplementedError("size is not implemented")
    def _pdf(self):
        raise NotImplementedError("pdf is not implemented")
    def _logpdf(self, x):
        raise NotImplementedError("log_pdf is not implemented")

    def _sample(self):
        raise NotImplementedError("sample is not implemented")



from collections import defaultdict
from transformers import pipeline

class MoralDetector:

    def __init__(self):
        self.lst_emotion = [f"This sentence expresses {ele}" for ele in
                            ['anger', 'fear', 'joy', 'surprise', 'sadness', 'disgust']]
        self.lst_polarity = ['This sentence is negative', 'This sentence is positive']
        self.lst_values = ['fairness', 'cheating', 'care', 'harm', 'loyalty', 'betrayal', 'authority', 'subversion',
                           'purity', 'degradation', 'non-moral']
        self.theshold = 0.9
        self.model = pipeline('zero-shot-classification', model='microsoft/deberta-large-mnli', device='cpu',
                              multiclass=True)

    def execution(self, txt):
        emotion = self.model(txt, self.lst_emotion)
        emotion = f"{emotion['labels'][0]}" if emotion['scores'][0] >= self.theshold else ''
        polarity = self.model(txt, self.lst_polarity)
        polarity = f"{polarity['labels'][0]}" if polarity['scores'][0] >= self.theshold else ''

        polarity = polarity.replace("This sentence is", "").replace("positive", "positivity").replace('negative',
                                                                                                      'negativity').strip()
        if len(emotion) > 0 and len(polarity) > 0:
            info_emo = f'{emotion} and {polarity}'
        elif len(emotion) > 0 and len(polarity) == 0:
            info_emo = emotion
        elif len(polarity) > 0 and len(emotion) == 0:
            info_emo = f"This sentence expresses {polarity}."
        else:
            info_emo = ''

        res = defaultdict(list)
        for emo in self.lst_values:
            res[emo].append(self.model(f"{txt}. {info_emo}", emo)['scores'][0])

        res_ = {k: v[0] for k, v in res.items() if v[0] >= self.theshold}
        res_ = {k: v for k, v in sorted(res_.items(), key=lambda item: item[1], reverse=True)}
        res_ = {max(res, key=res.get): res[max(res, key=res.get)][0]} if len(res_) == 0 else res_
        res_ = {'non-moral': '-'} if 'non-moral' == max(res_, key=res_.get) else {k: v for k, v in res_.items() if
                                                                                  k != 'non-moral'}
        out = {'labels': list(res_.keys()), 'scores': list(res_.values())}
        return out

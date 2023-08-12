from collections import defaultdict
from transformers import pipeline
import flask
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class MoralDetector:

    def __init__(self):
        self.device = 'cpu'
        self.lst_emotion = [f"Questa frase esprime {ele}" for ele in
                            ['rabbia', 'paura', 'gioia', 'sorpresa', 'tristezza', 'disgusto']]
        self.lst_polarity = ['Questa frase è negativa', 'Questa frase è positiva']
        self.lst_values = ['equità', 'frode', 'cura', 'danno', 'lealtà', 'tradimento', 'autorità', 'ribellione',
                           'purezza', 'degradado']

        self.mapping = {'equità':'Fairness', 'frode':'Cheating', 'cura':'Care', 'danno':'Harm', 'lealtà':'Loyalty', 'tradimento':'Betrayal', 'autorità':'Loyalty', 'ribellione':'Subversion',
                           'purezza':'Purity', 'degradado':'Degradation'}
        self.model = AutoModelForSequenceClassification.from_pretrained('Jiva/xlm-roberta-large-it-mnli').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('Jiva/xlm-roberta-large-it-mnli')

    def model_execution (self,text,lst_hyp):
        results = defaultdict(list)
        for hyp in lst_hyp:
          sent_tok = self.tokenizer.encode(text, hyp, return_tensors='pt', truncation_strategy='do_not_truncate')
          logits = self.model(sent_tok.to(self.device))[0]
          entail_contradiction_logits = logits[:,[2,1]]
          probs = entail_contradiction_logits.softmax(dim=1)
          prob_label_is_true = probs[:,1]
          results[hyp].append(prob_label_is_true.tolist()[0])
        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

        main_hyp =  [(label,score) for label,score in results.items()][-1] if list(results.values())[-1][0] >= 0.5 else ''
        most_prob_hyp = [(label,score) for label,score in results.items() if score[0] >= 0.5] if main_hyp != '' else ''

        return main_hyp , most_prob_hyp

    def execution(self,txt):
        main_emotion = self.model_execution (txt,self.lst_emotion)[0]
        polarity = self.model_execution (txt,self.lst_polarity)[0]
        main_values = self.model_execution (f"{txt} {main_emotion} {polarity}".strip(),self.lst_values)[1]

        print_emotion = f"{main_emotion[0]} al {int(round(main_emotion[1][0],2)*100)}%" if main_emotion != '' else 'La frase non veicola emozioni.'
        print_polarity = f"{polarity[0]} al {int(round(polarity[1][0],2)*100)}%" if polarity != '' else 'La frase ha una polarità neutra.'
        print_values = 'I valori morali veicolati dalla frase sono:\n' + '\n'.join([f"- {self.mapping[value[0]]} ({value[0]}) al {int(round(value[1][0],2)*100)}%" for value in main_values[::-1]]) if main_values != '' else 'La frase non veicola un contenuto morale.'

        return f"{print_emotion}\n{print_polarity}\n{print_values}"



app = Flask(__name__)



@app.route('/predict', methods=['GET','POST'])
def infer_values():
    #detector = MoralDetector()
    txt = request.args['txt']
    return jsonify(detector.execution(txt))
    
@app.route('/', methods=['GET'])
def index():
    return '''Welcome!'''

if __name__ == '__main__':
    detector = MoralDetector()
    app.run(debug=True, host='0.0.0.0')
# Randomization matters How to defend against strong adversarial attacks

Code of the ICML paper : Randomization matters How to defend against strong adversarial attacks. Rafael Pinot, Raphael Ettedgui, Geovani Rizk, Yann Chevaleyre, Jamal Atif.

## Dependencies

Use the requirements.txt file

```
pip install -r requirements.txt
```

## Run

All the hyperparameters are in the config.json file.

To train a mixture

```
python train.py
```

To eval the mixture

```
python eval.py --adversary=[ADVERSARY] --alpha=[ALPHA]
```

Example : 

```
python eval.py --adversary=pgd --alpha=0.2
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
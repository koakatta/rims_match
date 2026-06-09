from django import forms


class NameForm(forms.Form):
    your_name = forms.CharField(label="Your name", max_length=100)
class UploadImageForm(forms.Form):
    title = forms.CharField(max_length=100, required=False)
    image = forms.ImageField()
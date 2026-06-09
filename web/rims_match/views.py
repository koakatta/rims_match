from django.conf import settings
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import NameForm, UploadImageForm
from django.core.files.storage import FileSystemStorage
import torch
import clip 
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from pathlib import Path
import json
# Create your views here.
def index(request):
    return render(request, "rims_match/upload.html")


def upload_pic(request):
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect("/thanks/")

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()

    return render(request, "name.html", {"form": form})

@csrf_exempt
def upload_image(request):
    uploaded_url = None
    uploaded_name = None

    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data["image"]
            similarity=match_all(image_file)

            return render(request, "rims_match/result.html",{"results": similarity})
    else:
        form = UploadImageForm()

    return render(
        request,
        "rims_match/upload.html",
        {"form": form, "uploaded_url": uploaded_url, "uploaded_name": uploaded_name},
    )
def image_match(imagetarget,imagebase:str):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess=clip.load("ViT-B/32",device=device)
    img1=preprocess(Image.open(imagetarget)).unsqueeze(0).to(device)
    img2=preprocess(Image.open(imagebase)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat1=model.encode_image(img1)
        feat2=model.encode_image(img2)
    feat1=feat1/feat1.norm(dim=-1,keepdim=True)
    feat2=feat2/feat2.norm(dim=-1,keepdim=True)
    similarity=(feat1@feat2.T).item()
    return similarity

def match_all(imagetarget):
    p=Path("rims")
    rims_list=[x for x in p.iterdir() if not x.is_dir()]
    result={}
    for i in rims_list:
        result[str(i)]=image_match(imagetarget,i)
    return result


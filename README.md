# satellites-internship

### Semantic segmenation of crop fields 

![Screenshot](demo/demo_field.png)


## Getting started

- Clone repo

```
git clone ...
cd satellites-internship
```

- Install dependencies

```
pip install -r requirements.txt
```

- Start training Unet3D(define in train.py path to dataset) 

```
python train.py
```

- Build web app image

```
docker build . --tag pastis
```

- Start web app

```
docker run -p 5000:5000 pastis
```

**_NOTE:_**  If you wanna use [Sentinel Hub API](https://services.sentinel-hub.com/oauth/auth?client_id=30cf1d69-af7e-4f3a-997d-0643d660a478&redirect_uri=https%3A%2F%2Fapps.sentinel-hub.com%2Fdashboard%2FoauthCallback.html&scope=&response_type=token&state=%252Faccount%252Fbilling) for inference you need to create/have account and paste CREDENTIALS(`instance_id`,`sh_client_id`,`sh_client_secret`) in app.py file code:
```
config = SHConfig()
config.instance_id = "<PUT YOUR CREDENTIALS HERE>"
config.sh_client_id = "<PUT YOUR CREDENTIALS HERE>"
config.sh_client_secret = "<PUT YOUR CREDENTIALS HERE>"
config.save()
```





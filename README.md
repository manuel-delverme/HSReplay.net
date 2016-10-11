# [HSReplay.net](https://hsreplay.net)

A website to upload and share your Hearthstone games.


## Technology overview

The full backend stack is written in Python 3.

* Web framework: [Django](https://www.djangoproject.com/)
* Replay viewer: [Joust](https://github.com/HearthSim/joust/)
* HSReplay implementation: [HSReplay](https://github.com/HearthSim/hsreplay)
* Hearthstone library: [python-hearthstone](https://github.com/HearthSim/python-hearthstone)


### Django libraries

* API: [Django REST Framework](http://www.django-rest-framework.org/)
* Authentication: [Django Allauth](https://github.com/pennersr/django-allauth)
* Storage backends: [Django-Storages](https://github.com/jschneier/django-storages)
* Short IDs: [ShortUUID](https://github.com/stochastic-technologies/shortuuid)


### Production stack

* Accounts: [Battle.net API](https://dev.battle.net/)
* Web server: [Caddy](https://caddyserver.com/)
* App server: [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/)
* Database: [PostgreSQL (RDS)](https://aws.amazon.com/rds/postgresql/)
* Hosting: [Amazon Web Services](https://aws.amazon.com/)
* Analytics: [InfluxDB](https://influxdata.com/)
* Exception tracking: [Sentry](https://getsentry.com/)
* CI: [Jenkins](https://jenkins.io/)

Replays are processed on [Amazon Lambda](https://aws.amazon.com/lambda/details/)
using the Python 2.7 runtime.


## Installation

Prerequisites:

- [Vagrant](https://vagrantup.com) must be installed
- Virtualbox must be installed in order for the default provider to work

Set up:

- Run `vagrant up` to download and provision the box
- Once it is up, run `./scripts/run.sh` to start the server

The django server will then be available on `localhost:8000`.
The API is available at `/api/v1/` and is browsable using the DRF interface.


## License

Copyright © HearthSim - All Rights Reserved

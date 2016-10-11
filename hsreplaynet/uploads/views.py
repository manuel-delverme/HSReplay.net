from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.generic import View
from hsreplaynet.games.models import GameReplay
from .models import UploadEvent


class UploadDetailView(View):
	def get(self, request, shortid):
		try:
			upload = UploadEvent.objects.get(shortid=shortid)
		except UploadEvent.DoesNotExist:
			replay = GameReplay.objects.find_by_short_id(shortid)
			if replay:
				return HttpResponseRedirect(replay.get_absolute_url())

			# It is possible the UploadEvent hasn't been created yet.
			upload = None

		else:
			if upload.game:
				return HttpResponseRedirect(upload.game.get_absolute_url())

		return render(request, "uploads/processing.html", {"upload": upload})

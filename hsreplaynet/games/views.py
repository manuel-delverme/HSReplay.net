from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import Http404
from django.shortcuts import render
from django.views.generic import View
from django.views.decorators.clickjacking import xframe_options_exempt
from .models import GameReplay


class MyReplaysView(LoginRequiredMixin, View):
	def get(self, request):
		replays = GameReplay.objects.live().filter(user=request.user).count()
		context = {"replays": replays}
		return render(request, "games/my_replays.html", context)


class ReplayDetailView(View):
	def get(self, request, id):
		replay = GameReplay.objects.find_by_short_id(id)
		if not replay:
			return Http404()

		# TODO: IP caching in redis
		replay.views += 1
		replay.save()

		players = replay.global_game.players.all()
		players = players.prefetch_related("deck_list", "deck_list__includes")

		baseurl = "%s://%s" % (request.scheme, request.get_host())
		return render(request, "games/replay_detail.html", {
			"replay": replay,
			"title": replay.pretty_name_spoilerfree,
			"canonical_url": baseurl + replay.get_absolute_url(),
			"players": players,
			"twitter_card": request.GET.get("twitter_card", "summary")
		})


class ReplayEmbedView(View):
	@xframe_options_exempt
	def get(self, request, id):
		replay = GameReplay.objects.find_by_short_id(id)
		if not replay:
			return Http404()
		return render(request, "games/replay_embed.html", {"replay": replay})

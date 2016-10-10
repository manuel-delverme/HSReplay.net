from django.shortcuts import render
from django.views.generic import View
from .models import Scenario
from hsreplaynet.features.decorators import view_requires_feature_access


class ScenarioListView(View):
	@view_requires_feature_access("scenarios")
	def get(self, request):
		context = {
			"scenarios": Scenario.objects.filter(adventure=10).all()
		}

		return render(request, "scenarios/scenarios_list.html", context)


class ScenarioDetailsView(View):
	@view_requires_feature_access("scenarios")
	def get(self, request, scenario_id):
		context = {
			"scenario": Scenario.objects.get(pk=scenario_id),
			"ai_deck_list": Scenario.ai_deck_list(scenario_id),
			"winning_decks": Scenario.winning_decks(scenario_id)
		}

		return render(request, "scenarios/scenario_details.html", context)

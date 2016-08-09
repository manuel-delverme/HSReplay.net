from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.generic import View
from hsreplaynet.uploads import processing
from .models import UploadEvent


class UploadDetailView(View):
	def get(self, request, shortid):
		try:
			upload = UploadEvent.objects.get(shortid=shortid)
		except UploadEvent.DoesNotExist:
			# It is possible the UploadEvent hasn't been created yet.
			upload = None
		else:
			if upload.game:
				return HttpResponseRedirect(upload.game.get_absolute_url())

		return render(request, "uploads/processing.html", {"upload": upload})


class UploadFailuresListView(LoginRequiredMixin, View):
	def get(self, request):
		failed_uploads = processing.list_all_failed_raw_log_uploads()
		return render(request, "uploads/failures.html", {"failures": failed_uploads})


def reprocess_failed_upload(request, shortid):
	results = []
	if request.user.is_staff:
		results = processing.requeue_failed_raw_single_upload_with_id(shortid)

	failed_uploads = processing.list_all_failed_raw_log_uploads()
	return render(request, "uploads/failures.html", {"failures": failed_uploads, "results": results})

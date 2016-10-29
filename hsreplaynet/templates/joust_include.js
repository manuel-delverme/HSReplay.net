{% load static %}
{% load web_extras %}

var joust_required = [
	"{% joust_static 'joust.css' %}",
	"{% joust_static 'joust.js' %}",
	"{% static 'scripts/joust-extra.js' %}",
	"{% static 'fonts/belwefs_extrabold_macroman/stylesheet.css' %}",
	"{% static 'fonts/franklingothicfs_mediumcondensed_macroman/stylesheet.css' %}",
];

$(document).ready(function() {
	$("#joust-lightbox").click(function(e) {
		if(e.target == document.getElementById("joust-lightbox")) {
			e.preventDefault();
			$("#joust-lightbox").hide();
		}
	});
	$(document).keypress(function (e) {
		if (e.keyCode == 27) {
			$("#joust-lightbox").hide();
		}
	});
	var joust_started = false;
	var joust_check = function() {
		if(joust_required.length) {
			var file = joust_required.shift();
			var tag = file.match(/\.css$/) ? "link" : "script";
			var element = document.createElement(tag);
			element.onload = function() {
				joust_check();
			};
			if(tag == "link") {
				element.href = file;
				element.rel = "stylesheet";
			}
			else {
				element.src = file;
			}
			document.getElementsByTagName("head")[0].appendChild(element);
		}
		else if(!joust_started) {
			joust_started = true;

			var shim = document.createElement("style");
			shim.innerText = "{% filter escapejs %}{% include 'games/svg-paths-shim.css' with svg='/static/svg-paths.svg' %}{% endfilter %}";
			document.getElementsByTagName("head")[0].appendChild(shim);

			Joust.launcher("joust-promo-container")
				.assets("{% joust_static 'assets/' %}")
				.cardArt("{% setting 'HEARTHSTONE_ART_URL' %}")
				.width("100%")
				.height("100%")
				.fromUrl("{{ featured_game.replay_xml.url|safe }}");
		}
	};
	var trigger_resize = function() {
		var event = new UIEvent("resize", {view: window});
		window.dispatchEvent(event);
	};
	$("#feat-joust-screenshot").click(function(e) {
		$("#joust-lightbox").fadeIn();
		joust_check();
		trigger_resize();
		e.preventDefault();
	});
});


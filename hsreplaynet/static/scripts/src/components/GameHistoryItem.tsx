import * as React from "react";
import {GlobalGamePlayer, ImageProps, CardArtProps} from "../interfaces";
import GameHistoryPlayer from "./GameHistoryPlayer";
import {PlayState, BnetGameType} from "../hearthstone";


interface GameHistoryItemProps extends ImageProps, CardArtProps, React.ClassAttributes<GameHistoryItem> {
	shortid: string;
	players: GlobalGamePlayer[];
	startTime: Date;
	endTime: Date;
	gameType: number;
	disconnected: boolean;
	turns: number;
	won: boolean;
	friendlyPlayer: number;
}

interface GameHistoryItemState {
}

function humanTime(seconds) {
	// TODO: use something better
	var days = Math.floor((seconds % 31536000) / 86400);
	var hours = Math.floor(((seconds % 31536000) % 86400) / 3600);
	var mins = Math.floor((((seconds % 31536000) % 86400) % 3600) / 60);
	if (days) {
		return days + " days";
	}
	if (hours) {
		return hours + " hours";
	}
	return mins + " minutes";
}

export default class GameHistoryItem extends React.Component<GameHistoryItemProps, GameHistoryItemState> {
	constructor(props: GameHistoryItemProps, context: any) {
		super(props, context);
	}

	getDuration(): string {
		var seconds = this.props.endTime.getTime() - this.props.startTime.getTime();
		return humanTime(seconds / 1000);
	}

	getPlayedTime(): string {
		var seconds = new Date().getTime() - this.props.endTime.getTime();
		return humanTime(seconds / 1000) + " ago";
	}

	getIcon(): JSX.Element {
		if (this.props.disconnected) {
			return <img src={STATIC_URL + "images/dc.png"} className="hsreplay-type" alt="Disconnected"/>;
		}
		switch (this.props.gameType) {
			case BnetGameType.BGT_ARENA:
				return this.getArenaIcon();
			case BnetGameType.BGT_RANKED_STANDARD:
			case BnetGameType.BGT_RANKED_WILD:
				return this.getRankedIcon();
			case BnetGameType.BGT_TAVERNBRAWL_1P_VERSUS_AI:
			case BnetGameType.BGT_TAVERNBRAWL_2P_COOP:
			case BnetGameType.BGT_TAVERNBRAWL_PVP:
				return <img src={STATIC_URL + "images/modeID_Brawl.png"} className="hsreplay-type" alt="Tavern Brawl"/>;
			case BnetGameType.BGT_CASUAL_STANDARD:
			case BnetGameType.BGT_CASUAL_WILD:
				return <img src={STATIC_URL + "images/casual.png"} className="hsreplay-type" alt="Casual"/>;
			default:
				return null;
		}
	}

	getArenaIcon(): JSX.Element {
		var player = this.getFriendlyPlayer();
		var wins = player ? player.wins + 1 : 1;
		return <img src={STATIC_URL + "images/arena-medals/Medal_Key_" + wins + ".png"} className="hsreplay-type" alt="Arena"/>;
	}

	getRankedIcon(): JSX.Element {
		var player = this.getFriendlyPlayer();
		if (player) {
			if (player.rank) {
				return <img src={STATIC_URL + "images/ranked-medals/Medal_Ranked_" + player.rank + ".png"} className="hsreplay-type" alt={"Ranked"}/>
			}
			if (player.legend_rank) {
				return <img src={STATIC_URL + "images/ranked-medals/Medal_Ranked_Legend.png"} className="hsreplay-type" alt={"Legend"}/>
			}
		}
		return null;
	}

	getIconInfo(): string {
		var player = this. getFriendlyPlayer();
		if (!player) {
			return null;
		}
		switch (this.props.gameType) {
			case BnetGameType.BGT_ARENA:
				return +player.wins + " - " + +player.losses;
			case BnetGameType.BGT_RANKED_STANDARD:
			case BnetGameType.BGT_RANKED_WILD:
				if (player.rank) {
					return "Rank " + player.rank;
				}
				if (player.legend_rank) {
					return "Rank " + player.legend_rank;
				}
				return "Ranked";
			case BnetGameType.BGT_TAVERNBRAWL_1P_VERSUS_AI:
			case BnetGameType.BGT_TAVERNBRAWL_2P_COOP:
			case BnetGameType.BGT_TAVERNBRAWL_PVP:
				return "Tavern Brawl"
			default:
				return null;
		}
	}

	getFriendlyPlayer(): GlobalGamePlayer {
			return this.props.friendlyPlayer && this.props.players.find(x => x.player_id == this.props.friendlyPlayer);
	}

	render(): JSX.Element {
		return (<div className="col-xs-12 col-sm-6 col-md-4 col-lg-3 game-history-item">
			<a href={"/replay/" + this.props.shortid} className={this.props.won ? "won" : "lost"}>
				<div className="hsreplay-involved">
					<img src={this.props.image("vs.png")} className="hsreplay-versus"/>
					{this.props.players.sort(x => x.player_id == this.props.friendlyPlayer ? -1 : 1).map((player: GlobalGamePlayer, i: number) => {
						return <GameHistoryPlayer
							cardArt={this.props.cardArt}
							name={player.name}
							heroId={player.hero_id}
							won={player.final_state == PlayState.WON}
						/>;
					})}
				</div>
				<div className="hsreplay-details">
					<dl>
						<dt>Played</dt>
						<dd>{this.getPlayedTime()}</dd>
						<dt>Duration</dt>
						<dd>{this.getDuration()}</dd>
						<dt>Turns</dt>
						<dd>{Math.floor(this.props.turns / 2)} turns</dd>
					</dl>
					<div>
						{this.getIcon()}
						<div>{this.getIconInfo()}</div>
					</div>
				</div>
			</a>
		</div>);
	}
}

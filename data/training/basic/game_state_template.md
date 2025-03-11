{
    "game_info": {
        "format": "Standard",
        "turn_number": 3,
        "active_player": "Player1",
        "priority_player": "Player1",
        "phase": "Main1",
        "step": "None",
        "turn_structure": {
            "untap": {
                "completed": true
            },
            "upkeep": {
                "completed": true
            },
            "draw": {
                "completed": true
            },
            "main1": {
                "completed": false,
                "current": true
            },
            "combat": {
                "completed": false,
                "begin_combat": {
                    "completed": false
                },
                "declare_attackers": {
                    "completed": false
                },
                "declare_blockers": {
                    "completed": false
                },
                "combat_damage": {
                    "completed": false
                },
                "end_combat": {
                    "completed": false
                }
            },
            "main2": {
                "completed": false
            },
            "end": {
                "completed": false,
                "end_step": {
                    "completed": false
                },
                "cleanup": {
                    "completed": false
                }
            }
        },
        "priority": {
            "current_player": "Player1",
            "players_passed": [],
            "full_round_passed": false
        },
        "stack": [
            {
                "stack_id": 1,
                "timestamp": 1678956421,
                "object_type": "spell",
                "name": "Lightning Bolt",
                "controller": "Player2",
                "targets": [
                    {
                        "name": "Llanowar Elves",
                        "controller": "Player1"
                    }
                ],
                "mana_cost": "{R}",
                "colors": [
                    "R"
                ],
                "color_identity": [
                    "R"
                ],
                "type_line": "Instant",
                "rules_text": "Lightning Bolt deals 3 damage to any target.",
                "resolved": false,
                "position_on_stack": 1,
                "triggered_by": "None"
            }
        ]
    },
    "players": [
        {
            "name": "Player1",
            "life_total": 20,
            "poison_counters": 0,
            "mana_pool": {
                "W": 0,
                "U": 0,
                "B": 0,
                "R": 0,
                "G": 1,
                "C": 0,
                "sources": [
                    {
                        "source": "Llanowar Elves",
                        "amount": "{G}",
                        "restrictions": []
                    }
                ]
            },
            "lands_played_this_turn": 1,
            "counters": {
                "experience": 0,
                "energy": 0,
                "loyalty": 0,
                "other": {}
            },
            "zones": {
                "hand": {
                    "size": 5,
                    "cards": [
                        {
                            "name": "Forest",
                            "mana_cost": "",
                            "mana_value": 0,
                            "colors": [],
                            "color_identity": [
                                "G"
                            ],
                            "type_line": "Land — Forest",
                            "rules_text": "({T}: Add {G}.)",
                            "abilities": [
                                {
                                    "type": "Mana",
                                    "cost": "{T}",
                                    "effect": "Add {G}",
                                    "usable": true,
                                    "source_rule": "Printed"
                                }
                            ],
                            "visible_to_opponent": false
                        },
                        {
                            "name": "Giant Growth",
                            "mana_cost": "{G}",
                            "mana_value": 1,
                            "colors": [
                                "G"
                            ],
                            "color_identity": [
                                "G"
                            ],
                            "type_line": "Instant",
                            "rules_text": "Target creature gets +3/+3 until end of turn.",
                            "visible_to_opponent": false
                        },
                        {
                            "name": "Grizzly Bears",
                            "mana_cost": "{1}{G}",
                            "mana_value": 2,
                            "colors": [
                                "G"
                            ],
                            "color_identity": [
                                "G"
                            ],
                            "type_line": "Creature — Bear",
                            "power": 2,
                            "toughness": 2,
                            "rules_text": "",
                            "abilities": [],
                            "visible_to_opponent": false
                        },
                        {
                            "name": "Rampant Growth",
                            "mana_cost": "{1}{G}",
                            "mana_value": 2,
                            "colors": [
                                "G"
                            ],
                            "color_identity": [
                                "G"
                            ],
                            "type_line": "Sorcery",
                            "rules_text": "Search your library for a basic land card, put that card onto the battlefield tapped, then shuffle.",
                            "visible_to_opponent": false
                        },
                        {
                            "name": "Naturalize",
                            "mana_cost": "{1}{G}",
                            "mana_value": 2,
                            "colors": [
                                "G"
                            ],
                            "color_identity": [
                                "G"
                            ],
                            "type_line": "Instant",
                            "rules_text": "Destroy target artifact or enchantment.",
                            "visible_to_opponent": false
                        }
                    ]
                },
                "battlefield": {
                    "permanents": [
                        {
                            "name": "Forest",
                            "type_line": "Land — Forest",
                            "rules_text": "({T}: Add {G}.)",
                            "status": {
                                "tapped": false,
                                "flipped": false,
                                "face_down": false,
                                "phased_out": false,
                                "summoning_sick": false,
                                "attacking": false,
                                "blocking": [],
                                "exerted": false,
                                "transformed": false,
                                "monstrous": false,
                                "renowned": false,
                                "adapted": false
                            },
                            "abilities": [
                                {
                                    "type": "Mana",
                                    "cost": "{T}",
                                    "effect": "Add {G}",
                                    "usable": true,
                                    "source_rule": "Printed"
                                }
                            ],
                            "controller": "Player1",
                            "owner": "Player1",
                            "timestamp": 1,
                            "attachments": [],
                            "counters": {
                                "+1/+1": 0,
                                "-1/-1": 0,
                                "charge": 0,
                                "loyalty": 0,
                                "other": {}
                            }
                        },
                        {
                            "name": "Forest",
                            "type_line": "Land — Forest",
                            "rules_text": "({T}: Add {G}.)",
                            "status": {
                                "tapped": true,
                                "flipped": false,
                                "face_down": false,
                                "phased_out": false,
                                "summoning_sick": false,
                                "attacking": false,
                                "blocking": [],
                                "exerted": false,
                                "transformed": false,
                                "monstrous": false,
                                "renowned": false,
                                "adapted": false
                            },
                            "abilities": [
                                {
                                    "type": "Mana",
                                    "cost": "{T}",
                                    "effect": "Add {G}",
                                    "usable": false,
                                    "source_rule": "Printed"
                                }
                            ],
                            "controller": "Player1",
                            "owner": "Player1",
                            "timestamp": 2,
                            "attachments": [],
                            "counters": {
                                "+1/+1": 0,
                                "-1/-1": 0,
                                "charge": 0,
                                "loyalty": 0,
                                "other": {}
                            }
                        },
                        {
                            "name": "Llanowar Elves",
                            "mana_cost": "{G}",
                            "mana_value": 1,
                            "colors": [
                                "G"
                            ],
                            "color_identity": [
                                "G"
                            ],
                            "type_line": "Creature — Elf Druid",
                            "power": 1,
                            "toughness": 1,
                            "rules_text": "{T}: Add {G}.",
                            "status": {
                                "tapped": false,
                                "flipped": false,
                                "face_down": false,
                                "phased_out": false,
                                "summoning_sick": true,
                                "attacking": false,
                                "blocking": [],
                                "exerted": false,
                                "transformed": false,
                                "monstrous": false,
                                "renowned": false,
                                "adapted": false
                            },
                            "abilities": [
                                {
                                    "type": "Mana",
                                    "cost": "{T}",
                                    "effect": "Add {G}",
                                    "usable": false,
                                    "source_rule": "Printed"
                                }
                            ],
                            "damage_marked": 0,
                            "controller": "Player1",
                            "owner": "Player1",
                            "timestamp": 3,
                            "attachments": [],
                            "counters": {
                                "+1/+1": 0,
                                "-1/-1": 0,
                                "charge": 0,
                                "loyalty": 0,
                                "other": {}
                            }
                        }
                    ]
                },
                "graveyard": {
                    "size": 1,
                    "cards": [
                        {
                            "name": "Birds of Paradise",
                            "mana_cost": "{G}",
                            "mana_value": 1,
                            "colors": [
                                "G"
                            ],
                            "color_identity": [
                                "G"
                            ],
                            "type_line": "Creature — Bird",
                            "power": 0,
                            "toughness": 1,
                            "rules_text": "{T}: Add one mana of any color.",
                            "abilities": [
                                {
                                    "type": "Mana",
                                    "cost": "{T}",
                                    "effect": "Add one mana of any color",
                                    "usable": false,
                                    "source_rule": "Printed"
                                }
                            ],
                            "position": 1,
                            "owner": "Player1"
                        }
                    ]
                },
                "exile": {
                    "size": 0,
                    "cards": [],
                    "exile_groups": {}
                },
                "library": {
                    "size": 52,
                    "top_revealed": false,
                    "known_cards": [],
                    "known_position": {}
                },
                "command": {
                    "size": 0,
                    "cards": []
                },
                "sideboard": {
                    "size": 15,
                    "cards": []
                }
            },
            "just_drew": false
        },
        {
            "name": "Player2",
            "life_total": 17,
            "poison_counters": 0,
            "mana_pool": {
                "W": 0,
                "U": 0,
                "B": 0,
                "R": 1,
                "G": 0,
                "C": 0,
                "sources": [
                    {
                        "source": "Mountain",
                        "amount": "{R}",
                        "restrictions": []
                    }
                ]
            },
            "lands_played_this_turn": 0,
            "counters": {
                "experience": 0,
                "energy": 0,
                "loyalty": 0,
                "other": {}
            },
            "zones": {
                "hand": {
                    "size": 6,
                    "known_cards": [],
                    "revealed_cards": [
                        {
                            "name": "Mountain",
                            "revealed_by": "Goblin Guide trigger",
                            "revealed_to": [
                                "Player1"
                            ]
                        }
                    ],
                    "inferred_cards": []
                },
                "battlefield": {
                    "permanents": [
                        {
                            "name": "Mountain",
                            "type_line": "Land — Mountain",
                            "rules_text": "({T}: Add {R}.)",
                            "status": {
                                "tapped": true,
                                "flipped": false,
                                "face_down": false,
                                "phased_out": false,
                                "summoning_sick": false,
                                "attacking": false,
                                "blocking": [],
                                "exerted": false,
                                "transformed": false,
                                "monstrous": false,
                                "renowned": false,
                                "adapted": false
                            },
                            "abilities": [
                                {
                                    "type": "Mana",
                                    "cost": "{T}",
                                    "effect": "Add {R}",
                                    "usable": false,
                                    "source_rule": "Printed"
                                }
                            ],
                            "controller": "Player2",
                            "owner": "Player2",
                            "timestamp": 4,
                            "attachments": [],
                            "counters": {
                                "+1/+1": 0,
                                "-1/-1": 0,
                                "charge": 0,
                                "loyalty": 0,
                                "other": {}
                            }
                        },
                        {
                            "name": "Mountain",
                            "type_line": "Land — Mountain",
                            "rules_text": "({T}: Add {R}.)",
                            "status": {
                                "tapped": false,
                                "flipped": false,
                                "face_down": false,
                                "phased_out": false,
                                "summoning_sick": false,
                                "attacking": false,
                                "blocking": [],
                                "exerted": false,
                                "transformed": false,
                                "monstrous": false,
                                "renowned": false,
                                "adapted": false
                            },
                            "abilities": [
                                {
                                    "type": "Mana",
                                    "cost": "{T}",
                                    "effect": "Add {R}",
                                    "usable": true,
                                    "source_rule": "Printed"
                                }
                            ],
                            "controller": "Player2",
                            "owner": "Player2",
                            "timestamp": 5,
                            "attachments": [],
                            "counters": {
                                "+1/+1": 0,
                                "-1/-1": 0,
                                "charge": 0,
                                "loyalty": 0,
                                "other": {}
                            }
                        },
                        {
                            "name": "Goblin Guide",
                            "mana_cost": "{R}",
                            "mana_value": 1,
                            "colors": [
                                "R"
                            ],
                            "color_identity": [
                                "R"
                            ],
                            "type_line": "Creature — Goblin Scout",
                            "power": 2,
                            "toughness": 2,
                            "rules_text": "Haste\nWhenever Goblin Guide attacks, defending player reveals the top card of their library. If it's a land card, that player puts that card into their hand.",
                            "status": {
                                "tapped": true,
                                "flipped": false,
                                "face_down": false,
                                "phased_out": false,
                                "summoning_sick": false,
                                "attacking": false,
                                "blocking": [],
                                "exerted": false,
                                "transformed": false,
                                "monstrous": false,
                                "renowned": false,
                                "adapted": false
                            },
                            "abilities": [
                                {
                                    "type": "Static",
                                    "effect": "Haste",
                                    "usable": true,
                                    "source_rule": "Printed"
                                },
                                {
                                    "type": "Triggered",
                                    "trigger_condition": "Whenever Goblin Guide attacks",
                                    "effect": "Defending player reveals the top card of their library. If it's a land card, that player puts that card into their hand.",
                                    "usable": true,
                                    "source_rule": "Printed"
                                }
                            ],
                            "damage_marked": 0,
                            "controller": "Player2",
                            "owner": "Player2",
                            "timestamp": 6,
                            "attachments": [],
                            "counters": {
                                "+1/+1": 0,
                                "-1/-1": 0,
                                "charge": 0,
                                "loyalty": 0,
                                "other": {}
                            }
                        }
                    ]
                },
                "graveyard": {
                    "size": 2,
                    "cards": [
                        {
                            "name": "Shock",
                            "mana_cost": "{R}",
                            "mana_value": 1,
                            "colors": [
                                "R"
                            ],
                            "color_identity": [
                                "R"
                            ],
                            "type_line": "Instant",
                            "rules_text": "Shock deals 2 damage to any target.",
                            "position": 1,
                            "owner": "Player2"
                        },
                        {
                            "name": "Monastery Swiftspear",
                            "mana_cost": "{R}",
                            "mana_value": 1,
                            "colors": [
                                "R"
                            ],
                            "color_identity": [
                                "R"
                            ],
                            "type_line": "Creature — Human Monk",
                            "power": 1,
                            "toughness": 2,
                            "rules_text": "Haste\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)",
                            "abilities": [
                                {
                                    "type": "Static",
                                    "effect": "Haste",
                                    "usable": false,
                                    "source_rule": "Printed"
                                },
                                {
                                    "type": "Triggered",
                                    "trigger_condition": "Whenever you cast a noncreature spell",
                                    "effect": "This creature gets +1/+1 until end of turn",
                                    "usable": false,
                                    "source_rule": "Printed"
                                }
                            ],
                            "position": 2,
                            "owner": "Player2"
                        }
                    ]
                },
                "exile": {
                    "size": 0,
                    "cards": [],
                    "exile_groups": {}
                },
                "library": {
                    "size": 52,
                    "top_revealed": false,
                    "known_cards": [],
                    "known_position": {}
                },
                "command": {
                    "size": 0,
                    "cards": []
                },
                "sideboard": {
                    "size": 15,
                    "cards": []
                }
            },
            "just_drew": false
        }
    ],
    "special_game_states": {
        "day_night_cycle": "NotActive",
        "monarch": "None",
        "initiative": "None",
        "city_blessing": [],
        "dungeon_progress": {},
        "emblems": [],
        "state_based_actions": {
            "checked": true,
            "pending_actions": [
                {
                    "type": "DestroyDueToLethalDamage",
                    "affected_permanent": {
                        "name": "Llanowar Elves",
                        "controller": "Player1"
                    },
                    "reason": "Will have 3 damage marked from Lightning Bolt"
                }
            ]
        },
        "continuous_effects": [
            {
                "source": "Game Rule",
                "description": "Summoning sickness applies to creatures that came under a player's control this turn.",
                "duration": "Indefinite",
                "layer": "None",
                "timestamp": 0
            }
        ],
        "replacement_effects": [],
        "turn_based_actions_pending": []
    },
    "perspective": "Player1",
    "game_context": {
        "turn_history": [
            {
                "turn_number": 1,
                "active_player": "Player1",
                "summary": "Player1 played Forest, cast Birds of Paradise"
            },
            {
                "turn_number": 2,
                "active_player": "Player2",
                "summary": "Player2 played Mountain, cast Goblin Guide and attacked for 2, then cast Monastery Swiftspear"
            },
            {
                "turn_number": 3,
                "active_player": "Player1",
                "summary": "Player1 played Forest, cast Llanowar Elves, Player2 cast Shock targeting Birds of Paradise"
            }
        ],
        "strategic_assessment": {
            "board_advantage": "Player2",
            "card_advantage": "Equal",
            "tempo_advantage": "Player2",
            "critical_decision_points": [
                "Player1 must decide how to respond to Lightning Bolt targeting their Llanowar Elves"
            ]
        }
    },
    "game_metadata": {
        "date_played": "2025-03-11",
        "format": "Standard",
        "event_type": "Tournament",
        "match_id": "abc123",
        "round_number": 1,
        "game_number": 2,
        "decks": {
            "Player1": "Mono-Green Stompy",
            "Player2": "Mono-Red Aggro"
        }
    }
}
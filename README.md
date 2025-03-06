# Hexcore

by _Vren, the Relentless_

> _"Amber eyes and purple glow, rose-armored scavenger, friend or foe."_

Greetings, traveller.

I am **Vren, the Relentless**, scavenger and master of marsh and mire. With my amber eyes reflecting the eerie purple glow of my magical orb, I tread paths few dare to follow. The Calamity Beasts leave destruction and opportunity in their wake, and it is there, amidst the broken remnants and fallen relics, that I've crafted something extraordinary—a mind of arcane power, unmatched in the ways of Magic: The Gathering.

Welcome to **Hexcore**, a project born from curiosity, ingenuity, and just a touch of marsh madness.

## What is Hexcore?

Not simply a contraption of metal and magic, but a learned companion who has consumed and understood every line of MTG lore, rule, and strategy I could scavenge. It is an intelligence capable of:

- **Analysing the Battlefield:** Offer it your scenario, and it will peer through the arcane lens, suggesting optimal moves with tactical precision.
- **Constructing Decks:** Whisper your preferred style or strategic wish, and watch as it swiftly assembles a deck to dominate your foes, tailored precisely to your playstyle and the shifting meta.
- **Resolving Rules Mysteries:** Never again argue over obscure interactions; Hexcore knows the comprehensive rules better than a judge, retrieving exact references swiftly and precisely.
- **Advanced Reasoning:** Trained rigorously in five distinct modes—REASON, EXPLAIN, TEACH, PREDICT, and RETROSPECT—it will not merely answer questions but unravel mysteries, teach novices, predict opponents' moves, and learn from past missteps.

## Who is Vren?

Ah, I suppose introductions are proper.

Clad in armor forged from pressed roses, resilient yet delicate, armed with weapons fashioned from twisted roots, my garb reflects both the fragility and harshness of life in these marshlands. A slate mask conceals my visage, ensuring mystery remains intact as I prowl among the living and dying alike. My gang, loyal ratfolk with numbers enough to populate a small village, follows my command—to fight, scavenge, and trade as circumstances demand.

I am the scavenger behind Calamity Beasts, the figure lurking in their shadows, collecting treasures overlooked by others. But understand this clearly—my intent is neither wholly altruistic nor purely sinister. When the beasts threaten communities, I establish markets to arm and armor defenders at unmatched prices. After all, recycled goods hold nearly complete profit margins. My motivations? Practicality with a dash of calculated benevolence.

## Getting Started

Follow these steps to awaken Hexcore from its digital slumber:

### Step 1: Summon Dependencies

Invoke Python 3.10+, PyTorch, Hugging Face Transformers, and PEFT into your environment. See `environment.yml` for the full list of components.

```bash
conda env create -f environment.yml
conda activate hexcore
```

### Step 2: Retrieve the Mind

Due to its formidable size and intellect, the model's essence resides separately. Secure these model weights from the provided source and store them in your local sanctum under `models/`.

### Step 3: Knowledge Sources

A meticulously crafted Knowledge Graph and Retrieval system are bundled by default. To update or reconstruct these from fresh card sets or rules updates, refer to `data/README.md`.

### Step 4: Awakening Ritual

With all elements assembled, run:

```bash
python serve.py
```

Hexcore will rise at `http://localhost:5000`, eager to dispense wisdom.

## Interacting with Hexcore

Speak plainly, yet be detailed. For deeper insight or explicit reasoning, begin your query with keywords such as `REASON:` or ask explicitly, "How did you reach that conclusion?". Hexcore appreciates clarity.

## Contributing to the Cause

My resources are vast, but even I value community. Contributions from fellow scavengers, strategists, and sorcerers are welcome. If you find errors, misjudgments, or have enhancements, summon them forth through an issue or pull request. Detailed instructions reside in `CONTRIBUTING.md`.

## License & Legalities

Hexcore thrives under the permissive MIT License. The MTG content employed here—card texts, rules, and similar—belongs rightfully to Wizards of the Coast, incorporated responsibly and respectfully under fair use for educational and enthusiast purposes. This creation remains unofficial, so wield its knowledge with wisdom and caution in official play.

To make your README more finished while acknowledging that technical details are still to come, I'd suggest expanding a few sections:

1. Add a "Current Status" section indicating it's in development
2. Expand the "Getting Started" section with placeholder instructions
3. Add a roadmap for future features
4. Include contact information for questions

Here's how these additions could look:

## Current Status

Hexcore is currently in early development. The core model architecture is being refined, and training on MTG datasets is underway. This README outlines the vision and planned functionality, with technical implementations to follow in the coming weeks.

## Project Roadmap

- **Phase 1 (Current)**: Model architecture design and initial training
- **Phase 2**: Fine-tuning on specialized MTG datasets and rule interactions
- **Phase 3**: Web interface development and API implementation
- **Phase 4**: Community testing and refinement
- **Phase 5**: Full release with comprehensive documentation

## Technical Details (Coming Soon)

While the foundation has been laid, specific implementation details including model architecture, training methodology, and performance metrics will be added as development progresses. The following components are planned:

- MoE architecture based on Mistral 7B
- Specialized routing for different MTG knowledge domains
- Fine-tuning methodology optimized for card interactions
- Evaluation framework against known MTG puzzles and scenarios

## Contact & Questions

For inquiries about Hexcore or potential collaborations, reach out through:

- GitHub Issues

## Acknowledgments

Gratitude to the open-source spell-weavers and the MTG community. Together, we've created an intelligence worthy of legend.

May your path be clear, your profits rich, and your encounters manageable—preferably without the interference of Calamity Beasts.

_Ever Relentless,_
**Vren**

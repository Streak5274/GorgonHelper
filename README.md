# Project Gorgon Item Helper

A browser-based inventory and character tool for [Project Gorgon](https://www.projectgorgon.com/).
Tracks your storage vaults, work orders, recipes, favors, quests, and more — across all your characters.

---

## Requirements

- **Python 3.8+** — to run the local server ([download](https://www.python.org/downloads/))
- **Chrome or Edge** — required for the File System Access API (live folder watching)

---

## Setup

### 1. Get the files

Clone the repository or download it as a ZIP and extract it anywhere you like:

```
git clone https://github.com/Streak5274/GorgonHelper.git
```

### 2. Start the server

Open a terminal in the folder and run:

```
py start_server.py
```

You should see:
```
GorgonHelper serving on http://localhost:3000
```

> Leave this terminal open while using the tool.

### 3. Open in Chrome or Edge

Navigate to **http://localhost:3000/GorgonHelper.html** in Chrome or Edge.

### 4. Watch your Reports folder

The game writes character data to:
```
C:\Users\<you>\AppData\LocalLow\Elder Game\Project Gorgon\Reports\
```

In the tool, go to **Settings** → click **📂 Watch Folder** → select that `Reports` folder.
The tool will automatically load your character files and keep them up to date.

### 5. Sync game data

Still in **Settings**, under **Game Data Files**, click **Sync All** to download the latest item, recipe, and NPC data from the Project Gorgon CDN.

> This only needs to be done once, and again after game updates.

---

## Exporting character data from the game

The tool reads `.json` files exported by the game itself.

In-game, open the **Social** window → **Status** tab → click **Export to file**.
This writes a `Character_<name>.json` file to your `Reports` folder.

The tool picks up new exports automatically as long as the folder is being watched.

---

## Features

| Tab | What it does |
|---|---|
| **Storage** | Search all items across vaults and characters. Filter by vault, show consolidation candidates. |
| **Work Orders** | Track work order progress with ingredient availability. Shows available work orders. |
| **Recipes** | Browse known and locked recipes with skill level, NPC trainer info, and ingredient sources. |
| **Vaults** | See vault slot usage per NPC, with favor level progress. |
| **Favors** | Track favor levels with all NPCs across characters. |
| **Quests** | Active quest objectives with item counts and vault locations. |
| **Maps** | Interactive area maps with NPC and landmark positions. |
| **Tools** | Gardening tracker, Safecracking helper, Survey tool. |

---

## Updating

The tool can update itself. In **Settings** → **Version** → click **Check for updates**.
If an update is available, a button will appear to pull it automatically via `git`.

Or manually:
```
git pull origin master
```

---

## Notes

- The tool works fully offline after the initial game data sync — no account or login needed.
- All data is stored locally in your browser (IndexedDB) and in the `Reports` folder.
- Chrome/Edge are required for folder watching. The page can be opened as a file directly, but live updates won't work without the server.

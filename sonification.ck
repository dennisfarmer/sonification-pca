
5.0 => float n_seconds;

462 => int seed;
Math.srandom(seed);

// parameterized based on "jazz" tag
// 5 => int max_degree;

50.0 => float min_cutoff;
700.0 => float max_cutoff;

60.0 => float min_bpm;
180.0 => float max_bpm;

// --------------------
// --------------------
// instrument definitions

SinOsc osc;
ADSR env;
LPF lpf;
Gain g;

SinOsc overdrive;
440.0 => overdrive.freq;
1 => overdrive.sync;
Gain overdrive_g;

SinOsc bass_osc;
ADSR bass_env;
LPF bass_lpf;
Gain bass_g;

800.0 => bass_lpf.freq;
1.2 => bass_lpf.Q;

bass_env.set(
    0.4::second,
    2::second,
    0.7,
    2.0::second
);
//0.25 => bass_g.gain;


WvOut w;
(me.arg(0), IO.INT24) => w.wavFilename;

osc => env => lpf => g => dac;
env => overdrive => overdrive_g => dac;
bass_osc => bass_env => bass_lpf => bass_g => dac;

dac => w => blackhole;

//<<< "record file: ", w.filename() >>>;
// --------------------
// --------------------

["rock", "pop", "alternative", "indie", "electronic", 
"female vocalists", "dance", "00s", "alternative rock", "jazz", 
"beautiful", "metal", "chillout", "male vocalists", "classic rock", 
"soul", "indie rock", "mellow", "electronica", "80s", 
"folk", "90s", "chill", "instrumental", "punk", 
"oldies", "blues", "hard rock", "ambient", "acoustic", 
"experimental", "female vocalist", "guitar", "hip-hop", "70s", 
"party", "country", "easy listening", "sexy", "catchy", 
"funk", "electro", "heavy metal", "progressive rock", "60s", 
"rnb", "indie pop", "sad", "house", "happy" ] @=> string musicnn_tags[];

float musicnn_values[50];

<<< "dest -", me.arg(0) >>>;
for (0 => int i; i < musicnn_tags.size(); i++){
    Std.atof(me.arg(i+1)) => musicnn_values[i];
    //<<< musicnn_tags[i], "-", musicnn_values[i] >>>;
}

fun float tag_to_value( string tag ) {
    for ( 0 => int i; i < musicnn_values.size(); i++ ) {
        if (musicnn_tags[i] == tag) {
            return musicnn_values[i];
        }
    }
    return -100.0;
}

0.0 => float val;

// make high numbers low, 
// make low numbers high
// (convert values negatively correlated with
//  desired parameter to be positively correlated)
fun float invert( float val ) {
    return 1 - val;
}



// 7 possible modes
[
    [0, 2, 4, 6, 7, 9, 11], // Lydian (brightest)
    [0, 2, 4, 5, 7, 9, 11], // Ionian
    [0, 2, 4, 5, 7, 9, 10], // Mixolydian
    [0, 2, 3, 5, 7, 9, 10], // Dorian
    [0, 2, 3, 5, 7, 8, 10], // Aeolian
    [0, 1, 3, 5, 7, 8, 10], // Phrygian
    [0, 1, 3, 5, 6, 8, 10]  // Locrian (darkest)
] @=> int modes[][];

[
    "Lydian",
    "Ionian",
    "Mixolydian",
    "Dorian",
    "Aeolian",
    "Phrygian",
    "Locrian"
] @=> string mode_labels[];

// 12 possible keys
[
    60, // C
    61, // C#
    62, // D
    63, // D#
    64, // E
    65, // F
    66, // F#
    67, // G
    68, // G#
    69, // A
    70, // A#
    71  // B
] @=> int keys[];


// easily gets desired notes
fun float noteFreq(
    int mode,     // mode index (0 brightest, 6 darkest)
    int degree,   // scale degree (0-6, specifies which note of selected mode)
    int key,      // key as midi note (C = 60)
    int octave    // octave offset 0
    ) {
        degree % 7 => degree;
        key + modes[mode][degree] + (12*octave) => int midiNote;
        return Std.mtof(midiNote);
    }

fun float map_value( float val, float min, float max) {
    return min + val * (max-min);
}
fun int map_idx( float val, int max_idx ) {
    (val * (max_idx+1)) $ int => int idx;
    if (idx == max_idx+1) {
        max_idx => idx;
    }
    return idx;
}


<<< "\nPrincipal Component 1\n   (-) acoustic / rock -----  electronic / digital (+)\n\n" >>>;
// --------------------
// filter cutoff mapping
// --------------------
// tag values (pos): electro, electronic, electronica
// (take average)
// higher value: more electronic (higher cutoff)
// lower value: more natural/acoustic    (lower cutoff)
// note: not using acoustic tag as negatively correlated value
//       because it seems to be captured by same pc that captures electronic
//       (trying to have a more differentiated sound as pc is adjusted)

(
    tag_to_value("electro") + 
    tag_to_value("electronic") + 
    tag_to_value("electronica") +
    2 * tag_to_value("sexy") + 
    2 * tag_to_value("rnb") + 
    2 * tag_to_value("dance")
) / 10 => val;
min_cutoff + (val * (max_cutoff - min_cutoff)) => lpf.freq;
1.0 => lpf.Q;

map_value(invert(val), 0.15, 0.75) => float random_gain;

<<< "Cutoff parameter: (electro+electronic+electronica + 2*sexy+2*rnb+2dance) / 10 =", val >>>;
<<< "(between range ", min_cutoff, max_cutoff, ")" >>>;
<<< "low is acoustic, high is electronic" >>>;
<<< "Filter Cutoff", lpf.freq() >>>;
<<< "-----------------------------" >>>;

<<< "Random noise gain (proportional to PC1)", val >>>;
<<< "more electronic -> more noise" >>>;
<<< "Random gain", random_gain >>>;
<<< "-----------------------------" >>>;


// --------------------
// overdrive mapping
// --------------------
(
    tag_to_value("rock") + 
    tag_to_value("classic rock") + 
    tag_to_value("progressive rock") + 
    tag_to_value("hard rock") + 
    tag_to_value("metal") + 
    tag_to_value("heavy metal") + 
    2 * tag_to_value("sexy") + 
    2 * tag_to_value("rnb")

) / 10 => val;
map_value(val, 0.0, 0.8) => overdrive_g.gain;
<<< "Overdrive parameter: average of the rocks + metals + 2sexy+2rnb", val >>>;
<<< "low is normal, high is overdriven" >>>;
<<< "Overdrive gain", overdrive_g.gain() >>>;
<<< "-----------------------------" >>>;



<<< "\nPrincipal Component 2\n   (-) fast / happy -----> slow / sad (+)\n\n" >>>;

// --------------------
// Mode Mapping:
// --------------------
// tag values (pos): sad
// tag values (neg): happy

// low is bright, high is dark
(
    tag_to_value("sad") +
    invert(tag_to_value("happy"))
) / 2 => val;

map_idx(val, 6) => int mode_idx;

<<< "Mode parameter: (sad + (1 - happy)) / 2 =", val >>>;
<<< "low is bright, high is dark" >>>;
<<< "Mode:", mode_labels[mode_idx] >>>;
<<< "-----------------------------" >>>;

// --------------------
// tempo mapping
// --------------------
// tag values (neg): ambient, easy listening, mellow, chill
// tag values (pos): punk
// (take average)
// higher value => faster tempo
// lower value => slower value
(
    invert(tag_to_value("ambient")) +
    invert(tag_to_value("easy listening")) +
    invert(tag_to_value("mellow")) +
    invert(tag_to_value("chill")) +
    invert(tag_to_value("sad")) +
    tag_to_value("punk") +
    tag_to_value("party") +
    tag_to_value("catchy")
) / 8 => val;
map_value(val, min_bpm, max_bpm) => float tempo_bpm;
<<< "Tempo parameter: (5 - (ambient+easy listening+mellow+chill+sad) + punk+party+catchy) / 8 =", val >>>;
<<< "(between range ", min_bpm, max_bpm, ")" >>>;
<<< "Tempo:", tempo_bpm >>>;
<<< "-----------------------------" >>>;



// --------------------
// envelope mapping
// --------------------
// tag values (pos): ambient, chill
(tag_to_value("ambient") + tag_to_value("chill")) / 2 => val;
env.set(
    map_value(val, 0.05, 0.75)::second,  // attack
    map_value(val, 0.05,  1)::second,  // decay
    map_value(val, 0.3,   0.95),         // sustain
    map_value(val, 0.05,  1.0)::second   // release
    );

<<< "Ambientness parameter: (ambient+chill) / 2 = ", val >>>;
<<< "low is short, high is sustained" >>>;
<<< "-----------------------------" >>>;


<<< "\nPrincipal Component 3\n many notes ----->  few notes\n\n" >>>;
// --------------------
// max degree mapping
// --------------------
// tag values (pos): jazz, soul, country, oldies
// 
// more jazzy - more notes to choose from
(
    tag_to_value("jazz") +
    tag_to_value("soul") + 
    tag_to_value("country") +
    tag_to_value("oldies")
) / 4 => val;
map_idx(val, 7) => int max_degree;

<<< "Max Degree parameter: (jass+soul+country+oldies) / 4 =", val >>>;
<<< "(between 0-7)" >>>;
<<< "low is one note, high is many notes" >>>;
<<< "Max degree", max_degree >>>;
<<< "-----------------------------" >>>;



fun void playback_loop( float tempo, int key_idx ) {
    (60.0 / tempo)::second => dur beat;
    beat / 8 => dur note_dur;

    0 => int degree;
    0 => int bass_degree;

    -1 => int octave;
    -2 => int bass_octave;

    // hacky way to get the bass to 
    // be half the speed
    // of the scale
    0 => int bass_dont_play;
    0 => int max_bass_dont_play;

    // same for scale
    0 => int scale_dont_play;
    2 => int max_scale_dont_play;

    keys[key_idx] => int key_midi;
    now => time start_time;
    n_seconds::second => dur fade_dur;
    // playback loop
    while (now - start_time < fade_dur) {

        1.0 - ((now - start_time) / fade_dur) => float fade;
        (fade * 0.3) * 0.8 => g.gain;
        (fade * 0.3) * random_gain => bass_g.gain;
        (fade * 0.3) * 0.2 => overdrive_g.gain;

        Math.random2(0, 6) => bass_degree;

        if (scale_dont_play == 0) {
            noteFreq(
                mode_idx,
                degree,
                key_midi,
                octave
            ) => osc.freq;
            env.keyOn();
            1 => scale_dont_play;
        }


        if (bass_dont_play == 0) {
            noteFreq(
                mode_idx,
                bass_degree,
                key_midi,
                bass_octave
            ) => bass_osc.freq;
            bass_env.keyOn();
            1 => bass_dont_play;
        }

        // ----------------------
        // ----------------------
        note_dur => now;
        // ----------------------
        // ----------------------

        if (bass_dont_play != 0) {
            if (bass_dont_play < max_bass_dont_play) {
                bass_dont_play + 1 => bass_dont_play;
            } else {
                bass_env.keyOff();
                0 => bass_dont_play;
            }
        }
        if (scale_dont_play != 0) {
            if (scale_dont_play < max_scale_dont_play) {
                scale_dont_play + 1 => scale_dont_play;
            } else {
                env.keyOff();
                0 => scale_dont_play;
            }
        }

        degree++;
        if (degree >= max_degree) {
            0 => degree;
            octave++;
            if (octave > 2) {
                0 => octave;
            }
        }

        bass_degree++;
        if (bass_degree >= max_degree) {
            0 => bass_degree;
            bass_octave++;
            if (bass_octave > 2) {
                0 => bass_octave;
            }
        }
    }

    0.0 => g.gain;
    0.0 => bass_g.gain;
}


0 => int key_idx;

playback_loop(tempo_bpm, key_idx);
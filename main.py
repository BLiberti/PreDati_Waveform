# main.py
# Versione: v30
# Data ultima modifica: 2026-09-08
# Descrizione: Efficienza basata su polarità - soglia applicata correttamente
# v30: rimossi carica, media, pedestal, amp_max, amp_min, dev_std
#      mantenuti: bck, bck_sig, vmax, vmin, tmax, tmin, q_bck, q_sig, q_tot
#                 eff_V, t_eff_V, eff_Q, eff_5rm, t_eff_5rm

import os
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EventoParams:
    """Classe per memorizzare parametri evento"""
    numero_evento: int
    HV: float
    temperatura: float
    pressione: float
    descrizione_run: str

@dataclass
class FormaOnda:
    """Classe per memorizzare forma d'onda singolo canale"""
    tempo: np.ndarray
    tensione: np.ndarray
    canale: int
    evento: int

class ConfigLoader:
    """Carica configurazione YAML"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self._parse_sottodirectory()
    
    def _parse_sottodirectory(self):
        """Parsa sottodirectory dal formato string"""
        sottodirectory_parsed = []
        
        for subdir_entry in self.config['sottodirectory']:
            if isinstance(subdir_entry, str):
                parts = subdir_entry.split()
                if len(parts) != 4:
                    raise ValueError(f"Formato non valido: '{subdir_entry}'")
                
                sottodirectory_parsed.append({
                    'nome': parts[0],
                    'HV': float(parts[1]),
                    'temperatura': float(parts[2]),
                    'pressione': float(parts[3])
                })
        
        self.config['sottodirectory'] = sottodirectory_parsed

class DataReader:
    """Legge file dati"""
    
    def __init__(self, config: ConfigLoader):
        self.formato = config.config['formato_dati']
    
    def estrai_canale_evento(self, filename: str) -> tuple:
        """Estrae canale e numero evento: CX--XX--YYYYY.txt"""
        nome_senza_ext = Path(filename).stem
        parts = nome_senza_ext.split('--')
        canale = int(parts[0][1:])
        evento = int(parts[2])
        return canale, evento
    
    def leggi_file_text(self, filepath: str) -> FormaOnda:
        """Legge file testo LeCroy"""
        filename = Path(filepath).name
        canale, evento = self.estrai_canale_evento(filename)
        dati = np.loadtxt(filepath, skiprows=5, delimiter=',')
        tempo = dati[:, 0]
        tensione = dati[:, 1]
        return FormaOnda(tempo, tensione, canale, evento)
    
    def leggi_file_binary(self, filepath: str) -> FormaOnda:
        """Legge file binario"""
        filename = Path(filepath).name
        canale, evento = self.estrai_canale_evento(filename)
        dati = np.fromfile(filepath, dtype=np.float32)
        n_punti = len(dati) // 2
        tempo = dati[:n_punti]
        tensione = dati[n_punti:]
        return FormaOnda(tempo, tensione, canale, evento)
    
    def leggi_file(self, filepath: str) -> FormaOnda:
        if self.formato == 'text':
            return self.leggi_file_text(filepath)
        elif self.formato == 'binary':
            return self.leggi_file_binary(filepath)
        else:
            raise ValueError(f"Formato non supportato: {self.formato}")

class AnalizzatoreFormaOnda:
    """Analizza le forme d'onda e estrae parametri"""
    
    def __init__(self, config: Dict):
        self.intervalli = config.get('intervalli_temporali', {})
        self.soglie = config.get('soglie_efficienza', {})
        self.impedenza = config.get('impedenza', {})
        self.polarita = config.get('polarita', {})
    
    def get_intervallo(self, canale: int) -> tuple:
        """Ritorna (t0, tinf, tsup) per il canale"""
        key = f"canale_{canale}"
        if key not in self.intervalli:
            return None, None, None
        
        params = self.intervalli[key]
        return params['t0'], params['tinf'], params['tsup']
    
    def get_soglie(self, canale: int) -> tuple:
        """Ritorna (soglia_V, soglia_Q) per il canale"""
        key = f"canale_{canale}"
        if key not in self.soglie:
            return None, None
        
        params = self.soglie[key]
        return params['soglia_V'], params['soglia_Q']
    
    def get_impedenza(self, canale: int) -> float:
        """Ritorna l'impedenza per il canale"""
        key = f"canale_{canale}"
        return self.impedenza.get(key, 1.0)
    
    def get_polarita(self, canale: int) -> str:
        """Ritorna la polarità per il canale"""
        key = f"canale_{canale}"
        return self.polarita.get(key, "positive")
    
    def calcola_parametri(self, forma_onda: FormaOnda) -> Dict:
        """Calcola parametri della forma d'onda nel suo intervallo temporale"""
        
        tempo = forma_onda.tempo
        tensione = forma_onda.tensione
        canale = forma_onda.canale
        
        # Ottieni intervalli per questo canale
        t0, tinf, tsup = self.get_intervallo(canale)
        soglia_V, soglia_Q = self.get_soglie(canale)
        Z = self.get_impedenza(canale)
        polarita = self.get_polarita(canale)
        
        # Se non ci sono intervalli, ritorna valori neutri
        if t0 is None or tinf is None or tsup is None:
            return {
                'bck': 0.0,
                'bck_sig': 0.0,
                'vmax': 0.0,
                'vmin': 0.0,
                'tmax': 0.0,
                'tmin': 0.0,
                'q_bck': 0.0,
                'q_sig': 0.0,
                'q_tot': 0.0,
                'eff_V': 0.0,
                't_eff_V': 0.0,
                'eff_Q': 0.0,
                'eff_5rm': 0.0,
                't_eff_5rm': 0.0,
            }
        
        # **BACKGROUND**: intervallo [t0, tinf]
        mask_bck = (tempo >= t0) & (tempo <= tinf)
        tensione_bck = tensione[mask_bck]
        tempo_bck = tempo[mask_bck]
        
        if len(tensione_bck) > 0:
            bck = float(np.mean(tensione_bck))
            bck_sig = float(np.std(tensione_bck))
        else:
            bck = 0.0
            bck_sig = 0.0
        
        # **CARICA BACKGROUND**: q_bck tra t0 e tinf (dopo sottrazione bck) / Z
        tensione_bck_corretta = tensione_bck - bck
        if len(tempo_bck) > 1:
            q_bck = float(np.trapz(tensione_bck_corretta, tempo_bck) / Z)
        else:
            q_bck = 0.0
        
        # **SEGNALE**: intervallo [tinf, tsup]
        mask_sig = (tempo >= tinf) & (tempo <= tsup)
        tensione_filtrata = tensione[mask_sig]
        tempo_filtrato = tempo[mask_sig]
        
        if len(tensione_filtrata) == 0:
            return {
                'bck': bck,
                'bck_sig': bck_sig,
                'vmax': 0.0,
                'vmin': 0.0,
                'tmax': 0.0,
                'tmin': 0.0,
                'q_bck': q_bck,
                'q_sig': 0.0,
                'q_tot': 0.0,
                'eff_V': 0.0,
                't_eff_V': 0.0,
                'eff_Q': 0.0,
                'eff_5rm': 0.0,
                't_eff_5rm': 0.0,
            }
        
        # **SOTTRAI BACKGROUND dal segnale**
        tensione_corretta = tensione_filtrata - bck
        
        # Calcola Vmax e Vmin sul segnale corretto (intervallo [tinf, tsup])
        vmax = float(np.max(tensione_corretta))
        vmin = float(np.min(tensione_corretta))
        
        # **TMAX e TMIN**: tempo associato a vmax e vmin
        idx_vmax = int(np.argmax(tensione_corretta))
        idx_vmin = int(np.argmin(tensione_corretta))
        tmax = float(tempo_filtrato[idx_vmax])
        tmin = float(tempo_filtrato[idx_vmin])
        
        # **CARICA SEGNALE**: q_sig tra tinf e tsup (dopo sottrazione bck) / Z
        q_sig = float(np.trapz(tensione_corretta, tempo_filtrato) / Z)
        
        # **CARICA TOTALE**: q_tot da tinf alla fine della finestra temporale (dopo sottrazione bck) / Z
        mask_tot = tempo >= tinf
        tensione_tot = tensione[mask_tot]
        tempo_tot = tempo[mask_tot]
        tensione_tot_corretta = tensione_tot - bck
        if len(tempo_tot) > 1:
            q_tot = float(np.trapz(tensione_tot_corretta, tempo_tot) / Z)
        else:
            q_tot = 0.0
        
        # **CALCOLA EFFICIENZE E TEMPI** (considero la polarità)
        if polarita == "negative":
            # Per segnali negativi: controlla se scende sotto -soglia_V
            # Es: se soglia_V = 0.200, controlla se vmin < -0.200
            eff_V = 1.0 if (vmin < -soglia_V) else 0.0
            t_eff_V = 0.0
            if eff_V > 0.5:
                idx_sorpasso = np.where(tensione_corretta < -soglia_V)[0]
                if len(idx_sorpasso) > 0:
                    t_eff_V = float(tempo_filtrato[idx_sorpasso[0]])
            
            # Carica: per segnali negativi, considera il valore assoluto
            eff_Q = 1.0 if (abs(q_sig) > soglia_Q) else 0.0
            
            # Efficienza 5*sigma
            soglia_5rm = 5.0 * bck_sig
            eff_5rm = 1.0 if (vmin < -soglia_5rm) else 0.0
            t_eff_5rm = 0.0
            if eff_5rm > 0.5:
                idx_sorpasso_5rm = np.where(tensione_corretta < -soglia_5rm)[0]
                if len(idx_sorpasso_5rm) > 0:
                    t_eff_5rm = float(tempo_filtrato[idx_sorpasso_5rm[0]])
        else:  # positive
            # Per segnali positivi: controlla se sale sopra soglia_V
            eff_V = 1.0 if (vmax > soglia_V) else 0.0
            t_eff_V = 0.0
            if eff_V > 0.5:
                idx_sorpasso = np.where(tensione_corretta > soglia_V)[0]
                if len(idx_sorpasso) > 0:
                    t_eff_V = float(tempo_filtrato[idx_sorpasso[0]])
            
            eff_Q = 1.0 if (q_sig > soglia_Q) else 0.0
            
            soglia_5rm = 5.0 * bck_sig
            eff_5rm = 1.0 if (vmax > soglia_5rm) else 0.0
            t_eff_5rm = 0.0
            if eff_5rm > 0.5:
                idx_sorpasso_5rm = np.where(tensione_corretta > soglia_5rm)[0]
                if len(idx_sorpasso_5rm) > 0:
                    t_eff_5rm = float(tempo_filtrato[idx_sorpasso_5rm[0]])
        
        return {
            'bck': bck,
            'bck_sig': bck_sig,
            'vmax': vmax,
            'vmin': vmin,
            'tmax': tmax,
            'tmin': tmin,
            'q_bck': q_bck,
            'q_sig': q_sig,
            'q_tot': q_tot,
            'eff_V': eff_V,
            't_eff_V': t_eff_V,
            'eff_Q': eff_Q,
            'eff_5rm': eff_5rm,
            't_eff_5rm': t_eff_5rm,
        }

class AnalizzatoreDati:
    """Orchestratore principale"""
    
    def __init__(self, config_path: str):
        self.config_loader = ConfigLoader(config_path)
        self.data_reader = DataReader(self.config_loader)
        self.analizzatore_onda = AnalizzatoreFormaOnda(self.config_loader.config)
        self.config = self.config_loader.config
        self.num_canali = self.config['num_canali']
        self.eventi = {}
    
    def scansiona_directory(self):
        """Scansiona subdirectory e legge file"""
        dir_principale = Path(self.config['directory_principale'])
        
        for subdir_config in self.config['sottodirectory']:
            nome_subdir = subdir_config['nome']
            subdir_path = dir_principale / nome_subdir
            
            if not subdir_path.exists():
                print(f"⚠️  Subdirectory non trovata: {subdir_path}")
                continue
            
            print(f"📂 Scansione: {nome_subdir}")
            self._processa_subdirectory(subdir_path, subdir_config)
    
    def _processa_subdirectory(self, subdir_path: Path, subdir_config: Dict):
        """Processa una singola subdirectory"""
        
        nome_run = subdir_config['nome']
        
        for file_path in subdir_path.glob('*.txt'):
            try:
                forma_onda = self.data_reader.leggi_file(str(file_path))
                evento_id_locale = forma_onda.evento
                canale_id = forma_onda.canale
                
                evento_id_globale = f"{nome_run}_{evento_id_locale:03d}"
                
                if evento_id_globale not in self.eventi:
                    self.eventi[evento_id_globale] = {
                        'canali': {},
                        'parametri': {},
                        'evento_locale': evento_id_locale,
                        'HV': subdir_config['HV'],
                        'temperatura': subdir_config['temperatura'],
                        'pressione': subdir_config['pressione'],
                        'descrizione_run': subdir_config['nome']
                    }
                
                # Salva la forma d'onda
                self.eventi[evento_id_globale]['canali'][canale_id] = forma_onda
                
                # Calcola i parametri
                parametri = self.analizzatore_onda.calcola_parametri(forma_onda)
                self.eventi[evento_id_globale]['parametri'][canale_id] = parametri
                
                print(f"  ✓ Evento {evento_id_globale}, Canale {canale_id}")
                print(f"    BCK: {parametri['bck']:.6f}±{parametri['bck_sig']:.6f}, Q_sig: {parametri['q_sig']:.3e} C, Eff_V: {parametri['eff_V']:.0f}")
                
            except Exception as e:
                print(f"  ✗ Errore: {file_path.name}: {e}")
    
    def scrivi_root(self):
        """Scrivi ROOT file usando PyROOT"""
        try:
            import ROOT
        except ImportError:
            print("❌ ROOT non installato. Installa con: pip install root")
            return
        
        dir_principale = Path(self.config['directory_principale'])
        output_path = dir_principale / self.config['root_output']
        
        root_file = ROOT.TFile(str(output_path), "RECREATE")
        tree = ROOT.TTree("events", "Dati forme d'onda con parametri e efficienze")
        
        GlobalNmb = np.array([0], dtype=np.int32)
        EvtNmb = np.array([0], dtype=np.int32)
        HV = np.array([0], dtype=np.float32)
        temperatura = np.array([0], dtype=np.float32)
        pressione = np.array([0], dtype=np.float32)
        npts = np.array([0], dtype=np.int32)
        
        tree.Branch("GlobalNmb", GlobalNmb, "GlobalNmb/I")
        tree.Branch("EvtNmb", EvtNmb, "EvtNmb/I")
        tree.Branch("HV", HV, "HV/F")
        tree.Branch("temperatura", temperatura, "temperatura/F")
        tree.Branch("pressione", pressione, "pressione/F")
        tree.Branch("npts", npts, "npts/I")
        tree.Branch("descrizione_run", ROOT.std.string(), "descrizione_run")
        
        Time_vec = ROOT.std.vector("float")()
        V_vec = {}
        for ch in range(1, self.num_canali + 1):
            V_vec[ch] = ROOT.std.vector("float")()
            tree.Branch(f"V{ch}", V_vec[ch])
        
        params = {}
        for ch in range(1, self.num_canali + 1):
            params[ch] = {
                'bck': np.array([0.0], dtype=np.float32),
                'bck_sig': np.array([0.0], dtype=np.float32),
                'vmax': np.array([0.0], dtype=np.float32),
                'vmin': np.array([0.0], dtype=np.float32),
                'tmax': np.array([0.0], dtype=np.float32),
                'tmin': np.array([0.0], dtype=np.float32),
                'q_bck': np.array([0.0], dtype=np.float32),
                'q_sig': np.array([0.0], dtype=np.float32),
                'q_tot': np.array([0.0], dtype=np.float32),
                'eff_V': np.array([0.0], dtype=np.float32),
                't_eff_V': np.array([0.0], dtype=np.float32),
                'eff_Q': np.array([0.0], dtype=np.float32),
                'eff_5rm': np.array([0.0], dtype=np.float32),
                't_eff_5rm': np.array([0.0], dtype=np.float32),
            }
            
            for param_name, param_array in params[ch].items():
                tree.Branch(f"{param_name}{ch}", param_array, f"{param_name}{ch}/F")
        
        tree.Branch("Time", Time_vec)
        
        global_counter = 0
        for evento_id in sorted(self.eventi.keys()):
            evt = self.eventi[evento_id]
            canali = evt['canali']
            parametri = evt['parametri']
            evt_locale = evt['evento_locale']
            
            GlobalNmb[0] = global_counter
            EvtNmb[0] = evt_locale
            HV[0] = evt['HV']
            temperatura[0] = evt['temperatura']
            pressione[0] = evt['pressione']
            
            tempo = None
            for cid in sorted(canali.keys()):
                tempo = canali[cid].tempo
                break
            
            Time_vec.clear()
            for ch in range(1, self.num_canali + 1):
                V_vec[ch].clear()
            
            if tempo is not None:
                npts[0] = len(tempo)
                for t in tempo:
                    Time_vec.push_back(float(t))
            
            for num_canale in range(1, self.num_canali + 1):
                if num_canale in canali:
                    tensione = canali[num_canale].tensione
                else:
                    tensione = np.zeros_like(tempo) if tempo is not None else np.array([])
                
                for v in tensione:
                    V_vec[num_canale].push_back(float(v))
                
                if num_canale in parametri:
                    p = parametri[num_canale]
                    params[num_canale]['bck'][0] = p['bck']
                    params[num_canale]['bck_sig'][0] = p['bck_sig']
                    params[num_canale]['vmax'][0] = p['vmax']
                    params[num_canale]['vmin'][0] = p['vmin']
                    params[num_canale]['tmax'][0] = p['tmax']
                    params[num_canale]['tmin'][0] = p['tmin']
                    params[num_canale]['q_bck'][0] = p['q_bck']
                    params[num_canale]['q_sig'][0] = p['q_sig']
                    params[num_canale]['q_tot'][0] = p['q_tot']
                    params[num_canale]['eff_V'][0] = p['eff_V']
                    params[num_canale]['t_eff_V'][0] = p['t_eff_V']
                    params[num_canale]['eff_Q'][0] = p['eff_Q']
                    params[num_canale]['eff_5rm'][0] = p['eff_5rm']
                    params[num_canale]['t_eff_5rm'][0] = p['t_eff_5rm']
            
            tree.Fill()
            global_counter += 1
        
        tree.Write()
        root_file.Close()
        
        print(f"\n✅ ROOT file salvato: {output_path}")
    
    def esegui_analisi_completa(self):
        """Esegui pipeline completa"""
        print("🚀 Inizio analisi...")
        self.scansiona_directory()
        print(f"\n📊 Trovati {len(self.eventi)} eventi")
        self.scrivi_root()

def main():
    config_path = "config.yaml"
    analizzatore = AnalizzatoreDati(config_path)
    analizzatore.esegui_analisi_completa()

if __name__ == "__main__":
    main()

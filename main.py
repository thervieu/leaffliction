import tkinter as tk
from tkinter import ttk


def main() ->int :

  # Create a Tkinter window
  window = tk.Tk()
  window.title("LeafFliction")
  window.geometry("800x600")
  
  # Création du widget Notebook
  notebook = ttk.Notebook(window)

  # Création des onglets
  onglet1 = tk.Frame(notebook)
  onglet2 = tk.Frame(notebook)
  onglet3 = tk.Frame(notebook)
  onglet4 = tk.Frame(notebook)

  # Ajout des onglets au widget Notebook
  notebook.add(onglet1, text="Analysis of the Data Set")
  notebook.add(onglet2, text="Data augmentation")
  notebook.add(onglet3, text="Image Transformation")
  notebook.add(onglet4, text="Classification")

  # Affichage du widget Notebook
  notebook.pack(expand=True, fill="both")


  # Start the main event loop
  window.mainloop()
  return 0

if __name__ == "__main__" :
  SystemExit(main()) 

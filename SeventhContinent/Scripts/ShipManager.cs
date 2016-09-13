using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using System.Collections.Generic;

public class ShipManager : MonoBehaviour {

    public string shipName;
    public int startGold;
    private int gold;
    public int workers;
    public int crates;
    public int capacity;
    public Text itemInfo;
    // use dictionary for cargo[name:quantity]
    [System.Serializable]
    private struct cargoInfo {
        public string cargoName;
        public int quantity;
    }
    [SerializeField] private cargoInfo[] cargos;


    // Use this for initialization
    void Start () {
        gold = startGold;
        itemInfo = gameObject.GetComponentInChildren<Text>();
    }
	
	// Update is called once per frame
	void Update () {
        string cargoText = "";
        foreach (cargoInfo cargo in cargos) {
            cargoText += "\n\t" + cargo.quantity + " " + cargo.cargoName;
        }
        itemInfo.text = shipName + "\nGold: " + gold + "g\nCapacity: " + crates + 
        " / " + capacity + cargoText + "\nWorker(s): " + workers;
    }
    
    public int getGold () {
        return gold;
    }

    public void goldMinus (int substract) {
        gold -= substract;
    }

    // returns index if found, -1 if not found
    public int findCargo(string find, int quantity) {
        int i = 0;
        foreach (cargoInfo cargo in cargos) {
            if (string.Equals(find, cargo.cargoName) && cargo.quantity >= quantity) {
                return i;
            }
            i++;
        }
        return -1;
    }

    public void sellCrates(string soldProduct, int soldQuantity, int goldEarned, int index) {
        if (index != -1) {
            cargos[index].quantity -= soldQuantity;
            gold += goldEarned;
        }
    }
}

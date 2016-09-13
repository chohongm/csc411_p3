using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;


public class MarketManager : MonoBehaviour {

    public Text marketInfo;
    public string buying;
    [SerializeField] private int quantity;
    [SerializeField] private int rate;
    [SerializeField] private int perCrates;
    public GameObject cart;

    //public string selling;
    private ShipManager shipmanager;

    // Use this for initialization
    void Start () {
        marketInfo = gameObject.GetComponentInChildren<Text>();
        shipmanager = GameObject.Find("Ship").GetComponent<ShipManager>();
        RandomizeAll();
    }
	
	// Update is called once per frame
	void Update () {
        marketInfo.text = "Buying: " + buying + "\nRate: " + rate + " g / " + perCrates + " crate(s)" + "\nQuantity: " + quantity + " crate(s)";
    }

    public void Randomize() {
        quantity = Random.Range(1, 12);
        perCrates = Random.Range(1, 6);
        rate = Random.Range(15, 25) * perCrates;
    }

    void RandomizeAll() {
        Transform markets = GameObject.Find("Markets").transform;
        foreach (Transform market in markets) {
            market.GetComponent<MarketManager>().Randomize();
        }
    }

    void MarketBuy() {
        int found = shipmanager.findCargo(buying, quantity);
        if (found != -1) {
            Debug.Log("pressed");
            shipmanager.sellCrates(buying, quantity, rate / perCrates * quantity, found);
            RandomizeAll();
        }
    }

    void OnMouseUp() {
        MarketBuy();
        spawnCart();
    } 

    void spawnCart() {
        GameObject spawned = (GameObject) Instantiate(cart, new Vector3(46.54f, 2.3f, 11.91f), transform.rotation);
    }
}

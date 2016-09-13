using UnityEngine;
using System.Collections;

public class SpawnCart : MonoBehaviour {

    public GameObject cartPrefab;
    public GameObject pathGO;

    // Use this for initialization
    void Start () {
        
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    public void Spawn() {
        GameObject cart = Instantiate(cartPrefab, transform.position, transform.rotation) as GameObject;
        cart.GetComponent<Cart>().pathGO = pathGO;
    }
}

using UnityEngine;
using System.Collections;

public class TextHover : MonoBehaviour {

    public GameObject textGO;
    private UnityEngine.UI.Text text;

    // Use this for initialization
    void Start () {
        //text = this.transform.FindChild("Text").gameObject.GetComponent<UnityEngine.UI.Text>();
        text = textGO.GetComponent<UnityEngine.UI.Text>();
        textGO.SetActive(false);
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    void OnMouseOver() {
        textGO.SetActive(true);
    }

    void OnMouseExit() {
        textGO.SetActive(false);
    }
}

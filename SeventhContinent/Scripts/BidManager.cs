using UnityEngine;
using System.Collections;

public class BidManager : MonoBehaviour {

    public string civName;
    public int desire;
    public int friendship;

    private float percentageRaise;
    private int fameDiff;

    public Transform seller;
    private int sellerFame;
    public int bidderFame;

    public int expectedPrice;
    public int baseChosenPrice;
    public UnityEngine.UI.Text merchantInfo;
    // Use this for initialization

    void Start() {
        setDesire();
        friendship = Random.Range(0, 100);
        bidderFame = Random.Range(0, 2000);
        merchantInfo = GetComponentInChildren<UnityEngine.UI.Text>();
        merchantInfo.text = "Civ Name" + "\nDesire : " + desire + "\nFriendship : " + friendship + "\nFame : " + bidderFame;
        sellerFame = seller.GetComponent<BidManager>().bidderFame;

        fameDiff = sellerFame - bidderFame;
        if (fameDiff > 1000) {
            fameDiff = 1000;
        } else if (fameDiff < 0) {
            fameDiff = 0;
        }
    }

    // Update is called once per frame
    void Update() {

    }

    public void setDesire() {
        // upto +1000 fameDiff increases initial desire upto 50.
        desire = Random.Range((fameDiff / 1000 * 50), 100);
    }

    public void updateDesire() {
        float[] ratioList = new float[6];
        ratioList[0] = 2;
        ratioList[1] = 1;
        ratioList[2] = 0.5f;
        ratioList[3] = 0.25f;
        ratioList[4] = 0.1111f;
        ratioList[5] = 0.0526f;
        
        percentageRaise = (baseChosenPrice - expectedPrice) / expectedPrice * 100;

        for (int i = 0; i < ratioList.Length; i++) {
            dropDesireByRatio(ratioList[i]);
        }
    }

    void dropDesireByRatio(float ratio) {

        float chanceDesireDrop = percentageRaise * ratio;
        int randInt = Random.Range(1, 100);

        if (chanceDesireDrop >= 100) {
            desire -= 10;
        } else if (chanceDesireDrop >= 90) {
            if (randInt >= 10) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 80) {
            if (randInt >= 20) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 70) {
            if (randInt >= 30) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 60) {
            if (randInt >= 40) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 50) {
            if (randInt >= 50) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 40) {
            if (randInt >= 60) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 30) {
            if (randInt >= 70) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 20) {
            if (randInt >= 80) {
                desire -= 10;
            }
            return;
        } else if (chanceDesireDrop >= 10) {
            if (randInt >= 90) {
                desire -= 10;
            }
            return;
        }
    }
}
